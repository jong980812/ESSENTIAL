import heapq
import os
import numpy as np
import math
import random
import sys
from typing import Iterable, Optional
import torch
from mixup import Mixup
from timm.utils import accuracy, ModelEma
import utils
from scipy.special import softmax
from optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
import time,json
import datetime
from utils import NativeScalerWithGradNormCount as NativeScaler
from pathlib import Path
from utils import print_matrix_with_aligned_averages
from utils import unfreeze_block

def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, 
                    class_mask=None, args = None,loss_scaler=None, inference = False):


    train_stats = {}
    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    rehearsal_stats = {}
    acc_list = []
    for task_id in range(args.num_tasks):
        # if task_id<1:
            # continue
        # SSv2 초반 epoch을 위해 만들어놓았지만, 현재 사용 안함.
        if task_id == 0 and args.data_set == "SSV2":
            warmup_epochs,epochs = args.warmup_epochs,args.epochs
        else:
            warmup_epochs,epochs = args.warmup_epochs,args.epochs
            
        print(f'task {task_id+1}/{args.num_tasks}')
        start_time = time.time()
        
        #lr scehdule
        total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
        num_training_steps_per_epoch = len(data_loader[task_id]['train'].dataset) // total_batch_size
        print("Use step level LR scheduler!")
        lr_schedule_values = utils.cosine_scheduler(
            args.lr, args.min_lr, epochs, num_training_steps_per_epoch,
            warmup_epochs=warmup_epochs, warmup_steps=args.warmup_steps,
        )
        if args.weight_decay_end is None:
            args.weight_decay_end = args.weight_decay
        wd_schedule_values = utils.cosine_scheduler(
            args.weight_decay, args.weight_decay_end, epochs, num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))


        print(f"Start task training for {epochs} epochs")

        if args.joint:
            if task_id< args.num_tasks-1:
                continue
                
        #TODO pick best model using validation
        max_accuracy = 0.0
        if task_id > 0:
            # reinit_optimizer
            if loss_scaler is None:
                optimizer_params = get_parameter_groups(
                    model_without_ddp, args.weight_decay, args.skip_weight_decay_list,
                    args.assigner.get_layer_id if args.assigner is not None else None,
                    args.assigner.get_scale if args.assigner is not None else None)
                model, optimizer, _, _ = args.ds_init(
                    args=args, model=model_without_ddp, model_parameters=optimizer_params, dist_init_required=not args.distributed,
                ) 
            else:

                if 'AIM' in args.model:
                    model.module.unfreeze(args.unfreeze_layers_after_base)
                    if args.model=='AIM_final':
                        model.module.freeze_all_associ()
                        model.module.unfreeze_current_associ(task_id)
                    model.to(args.device)
                
                optimizer = create_optimizer(
                args, model_without_ddp, skip_list=args.skip_weight_decay_list,
                get_num_layer=args.assigner.get_layer_id if args.assigner is not None else None, 
                get_layer_scale=args.assigner.get_scale if args.assigner is not None else None)
                loss_scaler = NativeScaler()
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'*******Task{task_id+1} params: {n_parameters}*******')

        #!************************ Traininig *************************************
        Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_epoch_start_checkpoint.pth'.format(task_id+1))

        state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                }
        utils.save_on_master(state_dict, checkpoint_path)
        for epoch in range(epochs): 
            if args.ssv2_first_finetune is not None and (task_id<1):
                break
            if args.joint or args.inference or args.debugging or args.no_training:
                break

            if args.distributed:
                data_loader[task_id]['train'].sampler.set_epoch(epoch)   
            header = f'Task {task_id+1}/{args.num_tasks}  Train Epoch: [{epoch} / {epochs}]'

            train_stats = train_one_epoch(model=model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,
                                        start_steps=epoch * num_training_steps_per_epoch,
                                        lr_schedule_values=lr_schedule_values, 
                                        wd_schedule_values=wd_schedule_values,
                                        num_training_steps_per_epoch=num_training_steps_per_epoch, 
                                        update_freq=args.update_freq, header= header,loss_scaler=loss_scaler,rehearsal=False
                                        )
            # save model per epoch
            if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.epochs:
                Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
                checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_epoch_{}_checkpoint.pth'.format(task_id+1, epoch+1))

                state_dict = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }
                utils.save_on_master(state_dict, checkpoint_path)

        #! Saving CLS TOken*****************************************************************************
        if args.get_frame_index:
            if utils.is_main_process():
                save_frame_index(model=model,data_loader=data_loader,device = device, task_id=task_id, class_mask = None, args = args)
            torch.distributed.barrier()
            data_loader[task_id]['rehearsal'].dataset.update_rehearsal(task_id,args)
            torch.distributed.barrier()
        #! *****************************************************************************
        
        #? ************************ Rehearsal *************************************
        if args.memory_size > 0 and not args.inference:# and task_id > 0:
            # model, unfreeze_list = unfreeze_block(model,['head','S_Adapter','MLP_Adapter'])
            # print(unfreeze_list)
            # print('Freeze for rehearsal')
                   # lr scehdule
            # model.module.unfreeze(args.unfreeze_layers_rehearsal)
            if args.model =='AIM_final':
                model.module.freeze_all_associ()
            optimizer = create_optimizer(
            args, model_without_ddp, skip_list=args.skip_weight_decay_list,
            get_num_layer=args.assigner.get_layer_id if args.assigner is not None else None, 
            get_layer_scale=args.assigner.get_scale if args.assigner is not None else None)
            loss_scaler = NativeScaler()
            total_batch_size = args.batch_size * args.update_freq * utils.get_world_size()
            num_training_steps_per_epoch = len(data_loader[task_id]['rehearsal'].dataset) // total_batch_size
            if num_training_steps_per_epoch ==0:
                num_training_steps_per_epoch=1
            print("Use step level LR scheduler!")
            lr_schedule_values = utils.cosine_scheduler(
                args.lr, args.min_lr, args.rehearsal_epochs, num_training_steps_per_epoch,
                warmup_epochs=warmup_epochs, warmup_steps=args.warmup_steps,
            )
            if args.weight_decay_end is None:
                args.weight_decay_end = args.weight_decay
            wd_schedule_values = utils.cosine_scheduler(
                args.weight_decay, args.weight_decay_end, args.rehearsal_epochs, num_training_steps_per_epoch)
            print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))


            print(f"Start rehearsal training for {args.rehearsal_epochs} epochs")
            print("Backbone Freeze")
            model.module.transformer.eval()
            model.module.conv1.eval()
            for epoch in range(args.rehearsal_epochs):
                # break
                if (args.data_set=='SSV2') and (task_id==0):# and (not args.use_aim_weight):
                    break 
                if args.distributed:
                    data_loader[task_id]['rehearsal'].sampler.set_epoch(epoch) 
                header = f'Task {task_id+1}/{args.num_tasks}  Rehearsal Epoch: [{epoch} / {args.rehearsal_epochs}]'
                rehearsal_stats = train_one_epoch(model=model, criterion=criterion, 
                                            data_loader=data_loader[task_id]['rehearsal'], optimizer=optimizer, 
                                            device=device, epoch=epoch, max_norm=args.clip_grad, 
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,
                                            start_steps=epoch * num_training_steps_per_epoch,
                                            lr_schedule_values=lr_schedule_values, 
                                            wd_schedule_values=wd_schedule_values,
                                            num_training_steps_per_epoch=num_training_steps_per_epoch, 
                                            update_freq=args.update_freq, header=header,loss_scaler=loss_scaler, rehearsal=True
                                            )
        #? *****************************************************************

        if args.no_valid:
            continue
        # continue
        # pre = torch.load(f'/data/jong980812/project/cil/videoCIL/NIPS/AIM_my/rehearsal_use_virtual/OUT/checkpoint/task{task_id+1}_checkpoint.pth')
        # model.module.load_state_dict(pre['model'],strict = True)
        val_stats = evaluate_till_now(model=model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args,test_mode=False)
        acc_list.append(val_stats['stat_matrix'].tolist())
        del val_stats['stat_matrix']
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }

            utils.save_on_master(state_dict, checkpoint_path)
    
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'rehearsal_{k}': v for k, v in rehearsal_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
            'epoch': epoch,}
        print(log_stats)
        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        

    if args.output_dir and utils.is_main_process():
        Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = os.path.join(args.output_dir, 'checkpoint/last_task{}_checkpoint.pth'.format(task_id+1))
        state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
    print('test')
    evaluate_till_now(model=model, data_loader=data_loader, device=device, 
                                task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args,test_mode=True)
    #! SSV2 는 필요함.
    if utils.is_main_process():
        print('Average Incremental Accuracy (VAL)')
        print_matrix_with_aligned_averages(acc_list,args.n_videos)

    torch.distributed.barrier()
    
def train_one_epoch(model: torch.nn.Module,  
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args = None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None,header=None,loss_scaler=None, rehearsal = False,
                    frame_making=False
                    ):

    model.train(set_training_mode)
    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = header
    print_freq = 10

    for data_iter_step, (samples, targets,vname,sample_task_id,selected_frame) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        indices = selected_frame.numpy()
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        samples = samples.to(device, non_blocking=True)
        # for class_index, classes in enumerate(class_mask):
        #     for c in classes:
        #         targets[targets == c] = class_index
        targets = targets.to(device, non_blocking=True)
        
        mask = None
        if class_mask is not None:
            if rehearsal:
                mask = []
                for i in range(task_id+1):
                    mask+=class_mask[i]
                    # mask.append(i)
            else:
                mask = class_mask[task_id]
        #!!!각 마스크 첫번째 값 빼줘서 target 범위를 0~ 으로 맞춰줌.
        if args.each_head and not rehearsal:
            first_class = mask[0]
            targets = targets-first_class
        #!!!
        if args.mixup_fn is not None:
            samples, targets = args.mixup_fn(samples, targets)
            
        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
            model, samples, targets, criterion,mask,task_id,args,device)
        else:
            with torch.cuda.amp.autocast():
                loss,frame_matching,token_matching,virtual_loss, output= train_class_batch(
                model, samples, targets, criterion,mask,task_id,sample_task_id,args,device,
                rehearsal,
                frame_making,indices)
        if loss is None:
            loss = torch.tensor(0.).to(device)
        loss_value = args.origin_weight*loss.item()
        if frame_matching is not None:
            frame_matching_value = args.frame_matching_weight*frame_matching.item()
            loss +=args.frame_matching_weight*frame_matching
            
        if token_matching is not None:
            token_matching_value = args.token_matching_weight*token_matching.item()
            loss +=args.token_matching_weight*token_matching
            
        if virtual_loss is not None:
            virtual_value = args.virtual_weight*virtual_loss.item()
            loss +=args.virtual_weight*virtual_loss

        # if order_loss is not None:
        #     loss+=order_loss
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()

            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if args.mixup_fn is None and not frame_making:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
            
            
        metric_logger.update(origin_loss=loss_value)
        metric_logger.update(virtual_loss=virtual_value) if virtual_loss is not None else None
        metric_logger.update(frame_matching=frame_matching_value) if frame_matching is not None else None 
        metric_logger.update(token_matching=token_matching_value) if token_matching is not None else None 
        # metric_logger.update(order=order_loss.item()) if order_loss is not None else None 
        # metric_logger.update(cls_aug_loss=cls_aug_loss.item()) if cls_aug_loss is not None else None 
        # metric_logger.update(debias=debias_loss.item()) if debias_loss is not None else None
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        metric_logger.update(grad_norm=grad_norm)
        


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_class_batch(model, samples, target, criterion,mask,task_id,sample_task_id,args,device,rehearsal,frame_making,selected_frame):
    
    # if args.each_head:
    # first_class = mask[0]
    # if args.order:
    outputs,frame_matching,token_matching = model(samples,train=True,task_id=task_id,sample_task_id=sample_task_id,
                                rehearsal = rehearsal,frame_making=frame_making,selected_frame=selected_frame) 
    # else:
    #     outputs,_= model(samples,train=True,task_id=task_id)

    if (mask is not None) and (not args.each_head) and (not args.cos): #! each head이면 안됌.
        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
        if len(outputs)==2:
            origin,virtual = outputs[0],outputs[1]
            if origin is not None:origin = origin.index_fill(dim=1, index=not_mask, value=float('-inf'))
            if virtual is not None:virtual = virtual.index_fill(dim=1, index=not_mask, value=float('-inf'))
        else:
            outputs = outputs.index_fill(dim=1, index=not_mask, value=float('-inf'))
    target = target#-first_class
    if args.cos:
        if len(outputs)==2:
            origin,virtual = outputs[0],outputs[1]
            loss = model.module.cos_loss(origin,target)
            loss_virtual =  model.module.cos_loss(virtual,target) if virtual is not None else None
        else:
            loss = model.module.cos_loss(outputs,target)
    else:
        if len(outputs)==2:
            loss = criterion(origin, target) if origin is not None else None
            loss_virtual=criterion(virtual, target) if virtual is not None else None
        else:
            loss = criterion(outputs, target)
    
    if args.debias:
        shuffled_indices = np.random.permutation(samples.shape[2])
        shuffled_inputs = samples[:, :,shuffled_indices]
        shuffled_outputs,_= model(shuffled_inputs,train=True,task_id=task_id)
        debias_loss = -torch.mean(torch.sum(torch.nn.functional.log_softmax(shuffled_outputs, dim=1) 
                * torch.ones(shuffled_outputs.shape[0], args.nb_classes, device=shuffled_inputs.device) 
                / args.nb_classes, dim=1))
        loss = loss + debias_loss
    if args.order:
        shuffled_indices = np.random.permutation(samples.shape[2])
        shuffled_inputs = samples[:, :,shuffled_indices]
        _,x_final = model(shuffled_inputs,train=True,task_id=task_id)
        B, T = x_final.shape[:2]
        t_label = torch.LongTensor(shuffled_indices).unsqueeze(0).repeat(B,1).to(args.device)
        order_loss = criterion(x_final.view(B*T, -1), t_label.view(-1))
        # TODO mixup
        # outputs = outputs.index_fill(dim=1, index=not_mask, value=float('-1e4'))
        # target = target.index_fill(dim=1, index=not_mask, value=int(0))
        # loss = loss + order_loss
  
        
    return loss,(frame_matching),token_matching,(loss_virtual), origin


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale





@torch.no_grad()
def evaluate(model: torch.nn.Module,  data_loader, 
            device, task_id=-1,all_mask = None, class_mask=None, args=None,header=None,til=False
            ,selector = None,get_all_frame=False):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    if get_all_frame:
        with torch.no_grad():
            for batch in metric_logger.log_every(data_loader, 10, header):
                videos = batch[0]
                target = batch[1]
                vname = batch[2]          
                videos = videos.to(device, non_blocking=True)
                    # for class_index, classes in enumerate(all_mask):
                    #     for c in classes:
                    #         target[target == c] = class_index
                target = target.to(device, non_blocking=True)

                # compute output

                with torch.cuda.amp.autocast():
                    frame_index,num_frames= model(videos,train=False,task_id=task_id,get_frame = True)
                print(f'Index: {frame_index},   {num_frames} {vname}')
            return None
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, header):
            videos = batch[0]
            target = batch[1]            
            videos = videos.to(device, non_blocking=True)
                # for class_index, classes in enumerate(all_mask):
                #     for c in classes:
                #         target[target == c] = class_index
            target = target.to(device, non_blocking=True)

            # compute output

            with torch.cuda.amp.autocast():
                # selection = selector(videos)
                # selection_results = selection.argmax(1)#(B)
                # if args.order:
                logits,_ = model(videos,train=False,task_id=task_id,inference = True)
                logits = logits[0]
                # else:
                # logits = model(videos,task_id,None) if til  else model(videos,train=False,task_id=task_id)
                       
                # if til:
                #     not_mask = np.setdiff1d(np.arange(args.nb_classes), class_mask)
                #     not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                #     logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=videos.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=videos.shape[0])
            
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,test_mode=False,get_all_frame = False):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss
    import random
    num_task = args.num_tasks
    for i in range(task_id+1):
        #! til은 adapter selection 하기 위함.
        mask = class_mask[i]
        if get_all_frame:
            header = 'Frame_index: [Task {}]'.format(i + 1)
            _ = evaluate(model=model, data_loader=data_loader[i]['for_cls'], 
                                device=device, task_id=task_id,all_mask=class_mask, class_mask=mask, args=args,header=header,til=False,selector = None,get_all_frame=get_all_frame)
        else:
            if test_mode:
                header = 'Test: [Task {}]'.format(i + 1)
                test_stats = evaluate(model=model, data_loader=data_loader[i]['test'], 
                                    device=device, task_id=task_id,all_mask=class_mask, class_mask=mask, args=args,header=header,til=False,selector = None)
            else:
                header = 'VAL: [Task {}]'.format(i + 1)
                test_stats = evaluate(model=model, data_loader=data_loader[i]['val'], 
                                    device=device, task_id=task_id,all_mask=class_mask, class_mask=mask, args=args,header=header,til=False,selector = None)

            stat_matrix[0, i] = test_stats['Acc@1']
            stat_matrix[1, i] = test_stats['Acc@5']
            stat_matrix[2, i] = test_stats['Loss']

            acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    test_stats['stat_matrix'] = stat_matrix[0]
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats
import copy
@torch.no_grad()
def save_frame_index_in_train(
                    model: torch.nn.Module, 
                    data_loader, 
                    device, 
                    task_id=-1, 
                    class_mask=None, 
                    args=None,):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Training Saving Frame Index: [Task {} train Loader]'.format(task_id + 1)
    memory_video_path = {'dataset_samples':[],'label_array':[],'selected_frame':[],'energy':[]}
    model.eval()
    new_dataset = data_loader.dataset
    # new_dataset.all_frames = True
    num_tasks = 1#utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rehearsal = torch.utils.data.DistributedSampler(new_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    new_dataloader = torch.utils.data.DataLoader(
                new_dataset, sampler=sampler_rehearsal,
                batch_size=1,#!
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False if len(new_dataset)<args.batch_size*utils.get_world_size() else True,#total batch가 rehearsal보다 크면 .
            )
    c=0
    for batch in metric_logger.log_every(new_dataloader, 100, header):
        c+=1
        videos = batch[0]
        target = batch[1]
        vname = batch[2]          
        videos = videos.to(device, non_blocking=True)
            # for class_index, classes in enumerate(all_mask):
            #     for c in classes:
            #         target[target == c] = class_index
        target = target.to(device, non_blocking=True)
        video_name = vname[0]+'.mp4'
        label = int(target.cpu())
        with torch.cuda.amp.autocast():
            frame_index,num_frames,logit= model(videos,train=False,task_id=task_id,get_frame = True)
        selected_index = frame_index.squeeze(0).squeeze(0).cpu().numpy()
        memory_video_path['dataset_samples'].append(video_name)
        memory_video_path['label_array'].append(label)
        memory_video_path['selected_frame'].append(selected_index.tolist())
        memory_video_path['energy'].append(round(logit[0,label].item(),4))
        # if c==10:
        #     break
    with open(os.path.join(args.output_dir,f'selected_frame_task_{task_id+1}.txt'), 'w') as file:
        json.dump(memory_video_path, file)
@torch.no_grad()
def save_rehearsal_from_selected_frame_in_training(args,task_id,class_mask):
    print("**************Sample Selection*******************")
    
    class_num = sum([len(mask) for mask in class_mask[:task_id+1]])
    from collections import defaultdict
    label_groups = defaultdict(list)
    if task_id == 0:
        with open(os.path.join(args.output_dir,f'selected_frame_task_{task_id+1}.txt'), 'r') as file:
            selected_frame = json.load(file)
        for idx, label in enumerate(selected_frame['label_array']):
            # 각 label에 해당하는 정보들을 하나의 딕셔너리로 묶어서 저장
            entry = {
                'sample': selected_frame['dataset_samples'][idx],
                'selected_frame': selected_frame['selected_frame'][idx],
                'energy': selected_frame['energy'][idx]
            }
            label_groups[label].append(entry)
        # 각 그룹별로 energy 기준 상위 N개를 추출합니다.
    else:
        for task in range(task_id+1):
            if task<task_id:
                with open(os.path.join(args.output_dir,f'selected_sample_frame_task_{task+1}.txt'), 'r') as file:
                    selected_frame = json.load(file)
            else:
                with open(os.path.join(args.output_dir,f'selected_frame_task_{task+1}.txt'), 'r') as file:
                    selected_frame = json.load(file)
            for idx, label in enumerate(selected_frame['label_array']):
                # 각 label에 해당하는 정보들을 하나의 딕셔너리로 묶어서 저장
                entry = {
                    'sample': selected_frame['dataset_samples'][idx],
                    'selected_frame': selected_frame['selected_frame'][idx],
                    'energy': selected_frame['energy'][idx]
                }
                label_groups[label].append(entry)
    
    new_memory = {}
    save_num = math.ceil(args.memory_size/class_num)
    print(f'Class num: {class_num}')
    print(f'Memory per class: {save_num}')
    for label, entries in label_groups.items():
        # 각 label에서 energy가 가장 높은 상위 N개를 선택
        top_entries = heapq.nlargest(save_num, entries, key=lambda x: x['energy'])
        if len(top_entries)<save_num:
            print("error")
        new_memory[label] = top_entries
    total_items = sum(len(items) for items in new_memory.values())
    if total_items>args.memory_size: print(f"{total_items} is over {args.memory_size}: Calibrating") 
    while total_items > args.memory_size:
        labels_with_full_save_num = [label for label in new_memory if len(new_memory[label]) == save_num]
        if not labels_with_full_save_num:
            total_items = sum(len(items) for items in new_memory.values())
            print(f"After_calibraing: {total_items}")
            break
        random_label = random.choice(labels_with_full_save_num)  # 무작위로 하나의 레이블 선택
        new_memory[random_label].pop()  # 선택된 레이블에서 마지막 항목 제거
        total_items -= 1
    print("\n\nSample num per Class:")
    for label, items in new_memory.items():
        print(f"Label {label}: {len(items)} samples")
    print(f'Total: {total_items}')
    dataset_samples = []
    label_array = []
    selected_frame = []
    energy = []
    # 각 레이블의 항목을 순회하며 리스트를 구성
    for label, items in new_memory.items():
        for item in items:
            dataset_samples.append(item['sample'])
            label_array.append(int(label))  # 'label1'에서 숫자만 추출
            selected_frame.append(item['selected_frame'])
            energy.append(item['energy'])
    # 모든 데이터를 하나의 딕셔너리로 결합
    structured_data = {
        'dataset_samples': dataset_samples,
        'label_array': label_array,
        'selected_frame': selected_frame,
        'energy': energy
    }
    with open(os.path.join(args.output_dir,f'selected_sample_frame_task_{task_id+1}.txt'), 'w') as file:
        json.dump(structured_data, file)
@torch.no_grad()
def save_frame_index(model: torch.nn.Module, 
                    data_loader, 
                    device, 
                    task_id=-1, 
                    class_mask=None, 
                    args=None,
                    ):
    '''
    task_id 들어오면 해당 txt읽어야함.
    '''
    if task_id!=0:
        with open(os.path.join(args.output_dir,f'rehearsal_task_{task_id}.txt'), 'r') as file:
            a = json.load(file)
            # pre_label_array = a['label_array']
            pre_dataset_samples = a['dataset_samples']
            pre_selected_frame = a['selected_frame']
            pre_task_id = a['samples_task_id']
    memory_video_path = {'dataset_samples':[],'label_array':[],'selected_frame':[],'samples_task_id':[]}
    model.eval()
    re_dataset = data_loader[task_id]['rehearsal'].dataset
    re_dataset.all_frames = True
    num_tasks = 1#utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_rehearsal = torch.utils.data.DistributedSampler(re_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    data_loader_rehearsal = torch.utils.data.DataLoader(
                re_dataset, sampler=sampler_rehearsal,
                batch_size=1,#!
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=False if len(re_dataset)<args.batch_size*utils.get_world_size() else True,#total batch가 rehearsal보다 크면 .
            )
    header = 'Saving Frame Index: [Task {} Rehearsal Loader]'.format(task_id + 1)
    metric_logger = utils.MetricLogger(delimiter="  ")
    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader_rehearsal, 100, header):
            videos = batch[0]
            target = batch[1]
            vname = batch[2]          
            videos = videos.to(device, non_blocking=True)
                # for class_index, classes in enumerate(all_mask):
                #     for c in classes:
                #         target[target == c] = class_index
            target = target.to(device, non_blocking=True)
            # compute output
            if args.data_set =='ActivityNet':
                video_name = vname
                total_frames = videos.shape[2]
                start_ratio= round(float(video_name['t_start'][0]) / float(video_name['video_duration'][0]),5)
                start_frame = int(start_ratio * total_frames) 
            else:
                video_name = vname[0]+('.mp4' if args.data_set!='UCF101' else '')
            label = int(target.cpu())
            if task_id!=0:
                if (video_name in pre_dataset_samples):
                    sample_index = pre_dataset_samples.index(video_name)
                    memory_video_path['dataset_samples'].append(video_name)
                    memory_video_path['label_array'].append(label)
                    memory_video_path['selected_frame'].append(pre_selected_frame[sample_index])
                    memory_video_path['samples_task_id'].append(pre_task_id[sample_index])
                    continue
            with torch.cuda.amp.autocast():
                if args.data_set =='Kinetics-400':
                    # frame_index = batch[3]
                    selected_index = np.array([i.item() for i in batch[4]])
                else:
                    frame_index,num_frames,logit= model(videos,train=False,task_id=task_id,get_frame = True)
                    selected_index = frame_index.squeeze(0).squeeze(0).cpu().numpy() #+ (start_frame if args.data_set =='ActivityNet' else 0)
            memory_video_path['dataset_samples'].append(video_name)
            memory_video_path['label_array'].append(label)
            memory_video_path['selected_frame'].append(selected_index.tolist())
            memory_video_path['samples_task_id'].append(task_id)
            # print(f'Index: {frame_index}, {num_frames} {vname}')
    with open(os.path.join(args.output_dir,f'rehearsal_task_{task_id+1}.txt'), 'w') as file:
            json.dump(memory_video_path, file)
    re_dataset.all_frames = False


@torch.no_grad()
def final_test(data_loader, model, device, file):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    final_result = []
    
    for batch in metric_logger.log_every(data_loader, 10, header):
        videos = batch[0]
        target = batch[1]
        ids = batch[2]
        chunk_nb = batch[3]
        split_nb = batch[4]
        videos = videos.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(videos)
            loss = criterion(output, target)

        for i in range(output.size(0)):
            string = "{} {} {} {} {}\n".format(ids[i], \
                                                str(output.data[i].cpu().numpy().tolist()), \
                                                str(int(target[i].cpu().numpy())), \
                                                str(int(chunk_nb[i].cpu().numpy())), \
                                                str(int(split_nb[i].cpu().numpy())))
            final_result.append(string)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = videos.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    if not os.path.exists(file):
        os.mknod(file)
    with open(file, 'w') as f:
        f.write("{}, {}\n".format(acc1, acc5))
        for line in final_result:
            f.write(line)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def merge(eval_path, num_tasks):
    dict_feats = {}
    dict_label = {}
    dict_pos = {}
    print("Reading individual output files")

    for x in range(num_tasks):
        file = os.path.join(eval_path, str(x) + '.txt')
        lines = open(file, 'r').readlines()[1:]
        for line in lines:
            line = line.strip()
            name = line.split('[')[0]
            label = line.split(']')[1].split(' ')[1]
            chunk_nb = line.split(']')[1].split(' ')[2]
            split_nb = line.split(']')[1].split(' ')[3]
            data = np.fromstring(line.split('[')[1].split(']')[0], dtype=np.float, sep=',')
            data = softmax(data)
            if not name in dict_feats:
                dict_feats[name] = []
                dict_label[name] = 0
                dict_pos[name] = []
            if chunk_nb + split_nb in dict_pos[name]:
                continue
            dict_feats[name].append(data)
            dict_pos[name].append(chunk_nb + split_nb)
            dict_label[name] = label
    print("Computing final results")

    input_lst = []
    print(len(dict_feats))
    for i, item in enumerate(dict_feats):
        input_lst.append([i, item, dict_feats[item], dict_label[item]])
    from multiprocessing import Pool
    p = Pool(64)
    ans = p.map(compute_video, input_lst)
    top1 = [x[1] for x in ans]
    top5 = [x[2] for x in ans]
    pred = [x[0] for x in ans]
    label = [x[3] for x in ans]
    final_top1 ,final_top5 = np.mean(top1), np.mean(top5)
    return final_top1*100 ,final_top5*100

def compute_video(lst):
    i, video_id, data, label = lst
    feat = [x for x in data]
    feat = np.mean(feat, axis=0)
    pred = np.argmax(feat)
    top1 = (int(pred) == int(label)) * 1.0
    top5 = (int(label) in np.argsort(-feat)[:5]) * 1.0
    return [pred, top1, top5, int(label)]






