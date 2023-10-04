import os, pickle
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
from kinetics import KineticsDataset, VideoMAE
from ssv2 import SSVideoClsDataset
from activitynet import ActivitynetDataset
import utils
def is_double_list(obj):
    if isinstance(obj, list):
        return all(isinstance(sub_obj, list) for sub_obj in obj)
    return False


def build_dataset(is_train, test_mode,anno_list,task_id, args,rehearsal=False):
    if args.data_set == 'Kinetics-400':

        if is_train is True:
            mode = 'train'
        elif test_mode is True:
            mode = 'test'
        else:  
            mode = 'validation'

        dataset = KineticsDataset(
            anno_list=anno_list,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
            task_id = task_id,
            rehearsal=False
            )
    
    elif args.data_set == 'SSV2':

        if is_train is True:
            mode = 'train'
        elif test_mode is True:
            mode = 'test'
        else:  
            mode = 'validation'

        dataset = SSVideoClsDataset(
            anno_list=anno_list,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)

    elif args.data_set == 'ActivityNet':

        if is_train is True:
            mode = 'train'
        elif test_mode is True:
            mode = 'test'
        else:  
            mode = 'validation'

        dataset = ActivitynetDataset(
            anno_list=anno_list,
            data_path='/data2/local_datasets/ActivityNet/videos',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
            task_id = task_id,
            rehearsal=rehearsal
            )
    
    else:
        raise NotImplementedError()

    return dataset

def build_continual_dataloader(args):
    dataloader = list()

    with open(args.anno_path, 'rb') as file:
        anno_list = pickle.load(file)
    classes_per_task = args.nb_classes // args.num_tasks
    args.classes_per_task = classes_per_task
    class_mask = [list(range(i * classes_per_task, (i + 1) * classes_per_task)) for i in range(args.num_tasks)]

    for i in range(args.num_tasks):
        dataset_train = build_dataset(is_train=True, test_mode=False, args=args,anno_list=anno_list['train'][i],task_id=i)
        dataset_val = build_dataset(is_train=False, test_mode=False, args=args,anno_list=anno_list['val'][i],task_id=i)
        
        # TODO Test views
        if args.data_set == 'ActivityNet':
            dataset_test = build_dataset(is_train=False, test_mode=False, args=args,anno_list=anno_list['val'][i],task_id=i) 
        else:
            dataset_test = build_dataset(is_train=False, test_mode=False, args=args,anno_list=anno_list['test'][i],task_id=i)
        torch.distributed.barrier()
        dataset_rehearsal = build_dataset(is_train=False, test_mode=False, args=args,anno_list=None,task_id=i,rehearsal = True)


        # Only consider multi-GPU situations
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        sampler_rehearsal = torch.utils.data.DistributedSampler(
            dataset_rehearsal, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_rehearsal = torch.utils.data.DataLoader(
            dataset_rehearsal, sampler=sampler_rehearsal,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )

        dataloader.append({'train': data_loader_train, 'val': data_loader_val, 'test':data_loader_test,'rehearsal': data_loader_rehearsal})
    return dataloader, class_mask
