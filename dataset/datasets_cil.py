import os, pickle
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
from kinetics_cil import KineticsDataset, VideoMAE
from ssv2_cil import SSVideoClsDataset
from activitynet_cil import ActivitynetDataset
from ucf101_cil import UCFVideoClsDataset
import utils
def is_double_list(obj):
    if isinstance(obj, list):
        return all(isinstance(sub_obj, list) for sub_obj in obj)
    return False


def build_dataset(is_train, test_mode,anno_list,task_id, args,rehearsal=False):
    '''
    num crop은 test mode 구현 안된 관계로 1로 하드코딩.
    data_path는 cluster별로 다르기 때문에 하드코딩. 돌리기전에 Check
    '''
    if args.data_set == 'Kinetics-400':
        if is_train is True:
            mode = 'train'
            data_path = os.path.join('/local_datasets/kinetics400_320p' ,'train')
        elif test_mode is True:
            mode = 'validation'
            data_path = os.path.join('/local_datasets/kinetics400_320p' ,'test')
        else:  
            mode = 'validation'
            data_path = os.path.join('/local_datasets/kinetics400_320p' ,'val')
        dataset = KineticsDataset(
            anno_list=anno_list,
            data_path=data_path,
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
            rehearsal=rehearsal
            )
    
    elif args.data_set == 'SSV2':
        if is_train is True:
            mode = 'train'
        elif test_mode is True:
            mode = 'test'
        else:  
            mode = 'validation'
        data_path ='/local_datasets/something-something/something-something-v2-mp4'
        dataset = SSVideoClsDataset(
            anno_list=anno_list,
            data_path=data_path,
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
    elif args.data_set == 'UCF101':

        if is_train is True:
            mode = 'train'
        elif test_mode is True:
            mode = 'test'
        else:  
            mode = 'validation'
        data_path = '/local_datasets/ucf101/videos'
        if not os.path.isdir(data_path):
            data_path = '/local_datasets/ucf101/videos'

        dataset = UCFVideoClsDataset(
            anno_list=anno_list,
            data_path=data_path,
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
    elif args.data_set == 'ActivityNet':

        if is_train is True:
            mode = 'train'
        elif test_mode is True:
            mode = 'test'
        else:  
            mode = 'validation'

        dataset = ActivitynetDataset(
            anno_list=anno_list,
            data_path='/local_datasets/Activitynet/videos_256/',
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

    args.classes_per_task = []
    n_vids_per_task = []
    for i in range(args.num_tasks):
        if i == 0:
            args.classes_per_task.append(len(anno_list['train'][i].keys()))
        else:
            args.classes_per_task.append(len(anno_list['train'][i].keys())+args.classes_per_task[i-1])
    class_mask  = []
    for i in range(args.num_tasks):
        if i == 0:
            class_mask.append(list(range(0,args.classes_per_task[i])))
        else:
            class_mask.append(list(range(args.classes_per_task[i-1],args.classes_per_task[i])))
#! video 개수 미리 세봄.
    for i in range(args.num_tasks):
        num = 0
        for k,v in anno_list['train'][i].items():
            num+=len(v)
        n_vids_per_task.append(num)
        
    class_name_list = []



    data_loader_rehearsal = None
    for i in range(args.num_tasks):
        dataset_train = build_dataset(is_train=True, test_mode=False, args=args,anno_list=anno_list['train'][i],task_id=i)
        dataset_val = build_dataset(is_train=False, test_mode=False, args=args,anno_list=anno_list['val'][i],task_id=i)
        if args.data_set == 'SSV2':
            args.n_videos.append(len(dataset_val))
        # TODO Test views

# # TODO Test views
        if args.data_set == 'ActivityNet' or args.data_set == 'SSV2':
            dataset_test = build_dataset(is_train=False, test_mode=False, args=args,anno_list=anno_list['val'][i],task_id=i) 
        else:
            dataset_test = build_dataset(is_train=False, test_mode=True, args=args,anno_list=anno_list['test'][i],task_id=i)
        if args.memory_size > 0:
            torch.distributed.barrier()
            dataset_rehearsal = build_dataset(is_train=True, test_mode=False, args=args,anno_list=None,task_id=i,rehearsal = True)
            torch.distributed.barrier()


        # Only consider multi-GPU situations
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()

        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        if args.memory_size > 0:
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
        if args.memory_size > 0:
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
    return dataloader, class_mask ,n_vids_per_task,class_name_list