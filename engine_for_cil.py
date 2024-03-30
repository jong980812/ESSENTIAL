import os
import numpy as np
import math
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


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, 
                    class_mask=None, args = None,loss_scaler=None):


    train_stats = {}
    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    rehearsal_stats = {}
    acc_list = []
    for task_id in range(args.num_tasks):
        # SSv2 초반 epoch을 위해 만들어놓았지만, 현재 사용 안함.
        if task_id == 0 and args.data_set == "SSV2":
            warmup_epochs,epochs = args.warmup_epochs//3,args.epochs//5
        else:
            warmup_epochs,epochs = args.warmup_epochs,args.epochs
            
        print(f'task {task_id+1}/{args.num_tasks}')
        start_time = time.time()
        
       # lr scehdule
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

                #! Adapter 에서 0번 태스크 이후 작동하는 함수들 따로 지정.
                if args.model == 'AIM_adapter_v2':
                    model.module.freeze()                  
                    model.module.transformer.add_adapters(mode=args.mode)
                    model.module.transformer.del_adapters()
                    model.to(args.device)
                    model_without_ddp = model.module
                elif args.model == 'AIM_adapter':
                    model.module.transformer.freeze_adapters()                
                    model.module.transformer.add_adapters(mode=args.mode)
                    model.module.transformer.del_adapters()
                    model.to(args.device)
                    model_without_ddp = model.module
                optimizer = create_optimizer(
                args, model_without_ddp, skip_list=args.skip_weight_decay_list,
                get_num_layer=args.assigner.get_layer_id if args.assigner is not None else None, 
                get_layer_scale=args.assigner.get_scale if args.assigner is not None else None)
                loss_scaler = NativeScaler()
         
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'*******Task{task_id+1} params: {n_parameters}*******')
        
        #!************************ Traininig *************************************
        for epoch in range(epochs): 
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
                                        update_freq=args.update_freq, header= header,loss_scaler=loss_scaler
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
        #!************************ Rehearsal *************************************
        if args.memory_size > 0:
                   # lr scehdule
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
            
            for epoch in range(args.rehearsal_epochs): 
                if args.distributed:
                    data_loader[task_id]['rehearsal'].sampler.set_epoch(epoch) 
                header = f'Task {task_id+1}/{args.num_tasks}  Rehearsal Epoch: [{epoch} / {args.rehearsal_epochs}]'
                rehearsal_stats = train_one_epoch(model=model, criterion=criterion, 
                                            data_loader=data_loader[task_id]['rehearsal'], optimizer=optimizer, 
                                            device=device, epoch=epoch, max_norm=args.clip_grad, 
                                            set_training_mode=True, task_id=task_id, class_mask=None, args=args,
                                            start_steps=epoch * num_training_steps_per_epoch,
                                            lr_schedule_values=lr_schedule_values, 
                                            wd_schedule_values=wd_schedule_values,
                                            num_training_steps_per_epoch=num_training_steps_per_epoch, 
                                            update_freq=args.update_freq, header=header,loss_scaler=loss_scaler
                                            )
                
    
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
                    num_training_steps_per_epoch=None, update_freq=None,header=None,loss_scaler=None
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

    for data_iter_step, (samples, targets, _, _) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if i ==0:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                        
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask = None
        if class_mask is not None:
            mask = class_mask[task_id]
       
        if args.mixup_fn is not None:
            samples, targets = args.mixup_fn(samples, targets)
            
        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
            model, samples, targets, criterion,mask,args,device)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                model, samples, targets, criterion,mask,args,device)

        loss_value = loss.item()

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

        if args.mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
            
            
        metric_logger.update(loss=loss_value)
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

def train_class_batch(model, samples, target, criterion,mask,args,device):
    

    outputs = model(samples)
    if mask is not None:
        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
        outputs = outputs.index_fill(dim=1, index=not_mask, value=float('-inf'))
        
        # TODO mixup
        # outputs = outputs.index_fill(dim=1, index=not_mask, value=float('-1e4'))
        # target = target.index_fill(dim=1, index=not_mask, value=int(0))
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale






@torch.no_grad()
def evaluate(model: torch.nn.Module,  data_loader, 
            device, task_id=-1, class_mask=None, args=None,header=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for batch in metric_logger.log_every(data_loader, 10, header):
            videos = batch[0]
            target = batch[1]            
            videos = videos.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            with torch.cuda.amp.autocast():
                logits = model(videos)
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
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,test_mode=False):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss
    print('eval')
    for i in range(task_id+1):
        if test_mode:
            header = 'Test: [Task {}]'.format(i + 1)
            test_stats = evaluate(model=model, data_loader=data_loader[i]['test'], 
                                device=device, task_id=i, class_mask=class_mask, args=args,header=header)
        else:
            header = 'VAL: [Task {}]'.format(i + 1)
            test_stats = evaluate(model=model, data_loader=data_loader[i]['val'], 
                                device=device, task_id=i, class_mask=class_mask, args=args,header=header)

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
