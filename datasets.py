import os, pickle
from torchvision import transforms
from transforms import *
from masking_generator import TubeMaskingGenerator
from kinetics import VideoClsDataset, VideoMAE
from ssv2 import SSVideoClsDataset

def is_double_list(obj):
    if isinstance(obj, list):
        return all(isinstance(sub_obj, list) for sub_obj in obj)
    return False



class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
        self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])
        self.transform = transforms.Compose([                            
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if args.mask_type == 'tube':
            self.masked_position_generator = TubeMaskingGenerator(
                args.window_size, args.mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.transform(images)
        return process_data, self.masked_position_generator()

    def __repr__(self):
        repr = "(DataAugmentationForVideoMAE,\n"
        repr += "  transform = %s,\n" % str(self.transform)
        repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
        repr += ")"
        return repr


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=None,
        setting=args.data_path,
        video_ext='mp4',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        lazy_init=False)
    print("Data Aug = %s" % str(transform))
    return dataset

def build_rehearsal_dataset(is_train, test_mode, args, current_task):
    mode = 'train'
    if args.data_set == 'Kinetics-400':
        dataset = VideoClsDataset(
                anno_list=args.memory_video_path,
                data_path='/local_datasets/kinetics400',
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
                current_task = -1
                )
    elif args.data_set == 'UCF101':
        dataset =  VideoClsDataset(
            anno_list=args.memory_video_path,
            data_path='/data2/local_datasets/UCF-101',
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
            current_task = -1
            )
    





    label_array = [int(i.split(' ')[-1]) for i in args.memory_video_path]
    current_classes = set(label_array)
    return dataset, {}, current_classes

def build_cil_dataset(is_train, test_mode, args, current_task):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.pkl')
            with open(anno_path, 'rb') as f:
                anno_path = pickle.load(f)
                anno_list = anno_path[current_task]
            nb_classes = 400
            numbers = [int(line.split()[-1]) for line in anno_list]
            current_classes = set(numbers)
        # valid 할때는 이전 task 까지 다 있어야됨
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.pkl') 
            with open(anno_path, 'rb') as f:
                anno_path = pickle.load(f)
                anno_list = anno_path[:current_task+1]
            nb_classes = 400 
            current_classes = []

        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.pkl') 
            with open(anno_path, 'rb') as f:
                anno_path = pickle.load(f)
                anno_list = anno_path[:current_task+1]
            nb_classes = 400 
            current_classes = []
            

        if is_double_list(anno_list) and mode is not 'train':     
            # valid and test인 경우 task dataset list로 반환
            dataset = []            
            for i in  anno_list:
                dataset.append(VideoClsDataset(
                anno_list=i,
                data_path='/local_datasets/kinetics400',
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
                args=args))

        else:     
            dataset = VideoClsDataset(
                anno_list=anno_list,
                data_path='/local_datasets/kinetics400',
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
                current_task = current_task
                )
        



    elif args.data_set == 'SSV2':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = SSVideoClsDataset(
            anno_path=anno_path,
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
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.pkl')
            with open(anno_path, 'rb') as f:
                anno_path = pickle.load(f)
                anno_list = anno_path[current_task]
            nb_classes = 101
            numbers = [int(line.split()[-1]) for line in anno_list]
            current_classes = set(numbers)
        # valid 할때는 이전 task 까지 다 있어야됨
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.pkl') 
            with open(anno_path, 'rb') as f:
                anno_path = pickle.load(f)
                anno_list = anno_path[:current_task+1]
            nb_classes = 101 
            current_classes = []

        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.pkl') 
            with open(anno_path, 'rb') as f:
                anno_path = pickle.load(f)
                anno_list = anno_path[:current_task+1]
            nb_classes = 101 
            current_classes = []
            

        if is_double_list(anno_list) and mode is not 'train':     
            # valid and test인 경우 task dataset list로 반환
            dataset = []            
            for i in  anno_list:
                dataset.append(
                        VideoClsDataset(
            anno_list=i,
            data_path='/data2/local_datasets/UCF-101',
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
            args=args)
                    
                )

        else:     
            #train
            dataset =  VideoClsDataset(
            anno_list=anno_list,
            data_path='/data2/local_datasets/UCF-101',
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
            current_task = current_task
            )
        
        
        
        
        
        
        
        
        
        
        
        
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
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
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    print(f"current task : {current_task+1} | {mode} | Number of the class = %d" % nb_classes)

    return dataset, int(nb_classes), current_classes


