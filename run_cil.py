import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger().setLevel(logging.WARNING)
import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from functools import partial
from pathlib import Path
from collections import OrderedDict

from mixup import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner

from datasets import build_cil_dataset,build_rehearsal_dataset
from engine_for_finetuning import train_one_epoch, validation_one_epoch, final_test, merge
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import  multiple_samples_collate
from utils import  get_args_cil
import utils
import modeling_finetune




def main(args, ds_init):
    utils.init_distributed_mode(args)

    if ds_init is not None:
        utils.create_ds_config(args)

    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    # random.seed(seed)

    cudnn.benchmark = True
    all_tasks = args.all_tasks
    model = None


    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        all_frames=args.num_frames * args.num_segments,
        tubelet_size=args.tubelet_size,
        fc_drop_rate=args.fc_drop_rate,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.attn_drop_rate,
        drop_block_rate=None,
        use_checkpoint=args.use_checkpoint,
        use_mean_pooling=args.use_mean_pooling,
        init_scale=args.init_scale,
    )

    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.num_frames // 2, args.input_size // patch_size[0], args.input_size // patch_size[1])
    args.patch_size = patch_size

    if args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # height (== width) for the checkpoint position embedding 
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size) )** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)

        model.to(device)
        model_ema = None
        # if args.model_ema:
        #     model_ema = ModelEma(
        #         model,
        #         decay=args.model_ema_decay,
        #         device='cpu' if args.model_ema_force_cpu else '',
        #         resume='')
        #     print("Using EMA with decay = %.8f" % args.model_ema_decay)

        model_without_ddp = model

        print("Model = %s" % str(model_without_ddp))

        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        if args.grad_from_block > 0:
            # if zero full finetuning
            print(f"freeze until {args.grad_from_block-1}")
            for m in model.parameters():
                m.requires_grad = False

            # Only finetune layers from block 'args.grad_from_block' onwards
            print("="*20)
            print("learnable params")
            for name, m in model.named_parameters():
                if 'block' in name:
                    block_num = int(name.split('.')[1])
                    if block_num >= args.grad_from_block:
                        print(name, end=" | ")
                        m.requires_grad = True
                elif 'fc_norm' in name or 'head' in name or 'cls_token' in name:
                    m.requires_grad = True
                    print(name, end=" | ")

            print("="*20)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        args.lr = args.lr * total_batch_size / 256
        args.min_lr = args.min_lr * total_batch_size / 256
        args.warmup_lr = args.warmup_lr * total_batch_size / 256
        print("LR = %.8f" % args.lr)
        print("Batch size = %d" % total_batch_size)
        print("Update frequent = %d" % args.update_freq)

        num_layers = model_without_ddp.get_num_layers()
        if args.layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(list(args.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None

        if assigner is not None:
            print("Assigned values = %s" % str(assigner.values))

        skip_weight_decay_list = model.no_weight_decay()
        print("Skip weight decay list: ", skip_weight_decay_list)

    if args.enable_deepspeed:
        loss_scaler = None
        optimizer_params = get_parameter_groups(
            model, args.weight_decay, skip_weight_decay_list,
            assigner.get_layer_id if assigner is not None else None,
            assigner.get_scale if assigner is not None else None)
        model, optimizer, _, _ = ds_init(
            args=args, model=model, model_parameters=optimizer_params, dist_init_required=not args.distributed,
        )

        print("model.gradient_accumulation_steps() = %d" % model.gradient_accumulation_steps())
        assert model.gradient_accumulation_steps() == args.update_freq
    else:
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        optimizer = create_optimizer(
            args, model_without_ddp, skip_list=skip_weight_decay_list,
            get_num_layer=assigner.get_layer_id if assigner is not None else None, 
            get_layer_scale=assigner.get_scale if assigner is not None else None)
        loss_scaler = NativeScaler()

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)


    if mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    args.auto_resume = False
    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler, model_ema=model_ema)


        

    print('start video cil learning!')
    for current_task in range(all_tasks):

        dataset_train, _,current_classes = build_cil_dataset(is_train=True, test_mode=False, args=args, current_task = current_task)

        dataset_rehearsal, _,current_classes = build_rehearsal_dataset(is_train=True, test_mode=False, args=args, current_task = current_task)

        dataset_val_lst, _,_ = build_cil_dataset(is_train=False, test_mode=False, args=args,current_task = current_task)
        dataset_test_lst, _ ,_= build_cil_dataset(is_train=False, test_mode=False, args=args,current_task = current_task)
        

        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
      

        sampler_rehearsal = torch.utils.data.DistributedSampler(
            dataset_rehearsal, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

      
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_rehearsal = %s" % str(sampler_rehearsal))
        print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                'This will slightly alter validation results as extra duplicate entries are added to achieve '
                'equal num of samples per-process.')
        data_loader_val_lst = []
        data_loader_test_lst = []


        for dataset_val,dataset_test in zip(dataset_val_lst,dataset_test_lst):
            data_loader_val_lst.append(torch.utils.data.DataLoader(
                dataset_val, sampler=torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False),
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ))

            data_loader_test_lst.append(torch.utils.data.DataLoader(
                dataset_test, sampler=torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False),
                batch_size=int(1.5 * args.batch_size),
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False
            ))
            
            
        # if dist eval false
        # sampler_val = torch.utils.data.SequentialSampler(dataset_val)

        if global_rank == 0 and args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
        else:
            log_writer = None

        if args.num_sample > 1:
            collate_func = partial(multiple_samples_collate, fold=False)
        else:
            collate_func = None

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=collate_func,
        )
        
        data_loader_rehearsal = torch.utils.data.DataLoader(
            dataset_rehearsal, sampler=sampler_rehearsal,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
            collate_fn=collate_func,
        )

        num_training_steps_per_epoch = len(dataset_train) // total_batch_size
        print("Number of training examples = %d" % len(dataset_train))
        print("Number of training training per epoch = %d" % num_training_steps_per_epoch)

        num_rehearsal_training_steps_per_epoch = len(dataset_rehearsal) // total_batch_size
        print("Number of rehearsal training examples = %d" % len(dataset_rehearsal))
        print("Number of rehearsal training training per epoch = %d" % num_rehearsal_training_steps_per_epoch)


        print("Use step level LR scheduler!")
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

        print("Use step level LR scheduler!")
        rehearsal_lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, args.rehearsal_epochs, num_rehearsal_training_steps_per_epoch,
            warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
        )
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        rehearsal_wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, args.rehearsal_epochs, num_rehearsal_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(rehearsal_wd_schedule_values), min(rehearsal_wd_schedule_values)))



        print('='*50)
        print('='*50)
        print('='*50)
        print('='*50)
        print(f'Current task index : {current_task+1}/{all_tasks}')
        print('='*50)
        print('='*50)
        print('='*50)
        print('='*50)


        
        print(f"Start training for {args.epochs} epochs")
        start_time = time.time()
        max_accuracy = 0.0
        for epoch in range(args.start_epoch, args.epochs):

            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch * num_training_steps_per_epoch * args.update_freq)
                
            train_stats = train_one_epoch(
                model, criterion, data_loader_train, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
                log_writer=log_writer, start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values, wd_schedule_values=wd_schedule_values,
                num_training_steps_per_epoch=num_training_steps_per_epoch, update_freq=args.update_freq,current_classes=current_classes,header='TASK EPOCH'
            )
            # if args.output_dir and args.save_ckpt:
            #     if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
            #         utils.save_model(
            #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #             loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
        #     if (epoch + 1) % args.val_freq == 0 :
        #         test_stats = validation_one_epoch(data_loader_val_lst, model, device)
        #         print(f"[VAL-CURRENT-TASK][{epoch+1}] TOTAL Accuracy of the network on the  val videos: {test_stats['total_acc1']:.2f}%")

        #         if max_accuracy < test_stats["total_acc1"]:
        #             max_accuracy = test_stats["total_acc1"]
        #             if args.output_dir and args.save_ckpt:
        #                 utils.save_model(
        #                     args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
        #                     loss_scaler=loss_scaler, epoch="best_current", model_ema=model_ema)

        #         print(f'Max total accuracy: {max_accuracy:.2f}%')
                
        # checkpoint = torch.load(os.path.join(args.output_dir,'checkpoint-best_current','mp_rank_00_model_states.pt'), map_location='cpu')['module']
        # msg = model.module.load_state_dict(checkpoint)
        # print('load best current model')
        # print(msg)

        max_accuracy = 0.0
        print('-'*20)
        print('-'*20)
        print(f'start rehearsal training for {args.rehearsal_epochs} epochs ')
        for epoch in range(args.start_epoch, args.rehearsal_epochs):
            
            if args.distributed:
                data_loader_rehearsal.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch * num_rehearsal_training_steps_per_epoch * args.update_freq)
            train_stats = train_one_epoch(
                model, criterion, data_loader_rehearsal, optimizer,
                device, epoch, loss_scaler, args.clip_grad, model_ema, mixup_fn,
                log_writer=log_writer, start_steps=epoch * num_rehearsal_training_steps_per_epoch,
                lr_schedule_values=rehearsal_lr_schedule_values, wd_schedule_values=rehearsal_wd_schedule_values,
                num_training_steps_per_epoch=num_rehearsal_training_steps_per_epoch, update_freq=args.update_freq,current_classes=current_classes,header='Rehearsal EPOCH'
            )
            # if args.output_dir and args.save_ckpt:
            #     if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.rehearsal_epochs:
            #         utils.save_model(
            #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
            #             loss_scaler=loss_scaler, epoch=epoch, model_ema=model_ema)
            test_stats = validation_one_epoch(data_loader_val_lst, model, device)
            print(f"[VAL-REHEARSAL][{epoch+1}] TOTAL Accuracy of the network on the  val videos: {test_stats['total_acc1']:.2f}%")

            if max_accuracy < test_stats["total_acc1"]:
                max_accuracy = test_stats["total_acc1"]
                if args.output_dir and args.save_ckpt:
                    utils.save_model(
                        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch="best_rehearsal", model_ema=model_ema)

            print(f'Max total accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(val_acc1=test_stats['total_acc1'], head="perf", step=epoch)
                # log_writer.update(val_acc5=test_stats['acc5'], head="perf", step=epoch)
                # log_writer.update(val_loss=test_stats['loss'], head="perf", step=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            if args.output_dir and utils.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")



        checkpoint = torch.load(os.path.join(args.output_dir,'checkpoint-best_rehearsal','mp_rank_00_model_states.pt'), map_location='cpu')['module']
        msg = model.module.load_state_dict(checkpoint)
        print('load best rehearsal model')
        print(msg)

        preds_file = os.path.join(args.output_dir, str(global_rank) + '.txt')
        test_stats = validation_one_epoch(data_loader_test_lst, model, device)
        print(f"[TEST] Accuracy of the network on the  test videos: {test_stats['total_acc1']:.2f}%")

        #TODO test
        # test_stats = final_test(data_loader_test, model, device, preds_file)
        # torch.distributed.barrier()
        # if global_rank == 0:
        #     print("Start merging results...")
        #     final_top1 ,final_top5 = merge(args.output_dir, num_tasks)
        #     print(f"Accuracy of the network on the {len(dataset_test)} test videos: Top-1: {final_top1:.2f}%, Top-5: {final_top5:.2f}%")
        #     log_stats = {'Final top-1': final_top1,
        #                 'Final Top-5': final_top5}
        #     if args.output_dir and utils.is_main_process():
        #         with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        #             f.write(json.dumps(log_stats) + "\n")
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print(f'Current task index : {current_task}/{all_tasks-1}')
        print('Training time {}'.format(total_time_str))
        torch.distributed.barrier()

    print('training finished')


if __name__ == '__main__':
    opts, ds_init = get_args_cil()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts, ds_init)
