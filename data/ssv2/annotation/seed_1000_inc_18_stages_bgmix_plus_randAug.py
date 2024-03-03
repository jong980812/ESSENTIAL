import os

data_dir = os.environ['VIDEO_CIL_ROOT']

# base settings
gpu_ids = 4

# single gpu setting for traning
videos_per_gpu = 12
workers_per_gpu = 2
accumulate_grad_batches = 1

# single gpu setting for testing
testing_videos_per_gpu = 1
testing_workers_per_gpu = 2

work_dir = 'work_dirs/sth-sthv2-seed_1000_inc_18_stages'

task_splits = [[33, 28, 129, 145, 147, 15, 64, 95, 101, 167, 57, 152, 171, 66, 49, 165, 155, 110, 16, 107, 37, 102,
                118, 91, 39, 62, 84, 24, 149, 146, 154, 115, 93, 68, 22, 52, 120, 142, 80, 108, 3, 44, 130, 17, 97,
                143, 70, 103, 4, 132, 38, 137, 43, 126, 116, 133, 60, 98, 156, 32, 121, 8, 141, 161, 131, 23, 99, 74,
                34, 117, 83, 111, 136, 166, 158, 153, 46, 139, 124, 172, 122, 67, 164, 162], [127, 90, 76, 86, 54],
               [159, 27, 112, 85, 82], [119, 9, 160, 78, 19], [63, 13, 73, 150, 168], [48, 125, 104, 81, 25],
               [96, 65, 20, 56, 134], [53, 51, 35, 106, 10], [6, 41, 59, 77, 2], [18, 72, 157, 55, 140],
               [5, 12, 135, 29, 79], [47, 151, 169, 109, 31], [0, 123, 113, 163, 173], [144, 7, 100, 26, 21],
               [50, 75, 11, 69, 61], [14, 138, 114, 88, 30], [148, 58, 42, 36, 170], [105, 40, 45, 89, 128],
               [1, 92, 94, 71, 87]]

# select one of ['base', 'oracle', 'finetune']
methods = 'base'
starting_task = 0
ending_task = 18
use_nme_classifier = False
use_cbf = False
cbf_train_backbone = False
budget_size = 20
storing_methods = 'videos'
budget_type = 'class'
num_epochs_per_task = 50

"""
mode: bgmix_plus_randAug 
    bgmix_prob = 1 - randAug_prob

mode: randAug only
    randAug_prob = 2    # set any value >= 1.0

mode: bgmix only
    randAug_prob = -1   # set any value < 0.0
"""
randAug_prob = 0.75  # bgmix_prob = 0.25

# mmaction2 model config
# model settings
starting_num_classes = len(task_splits[0])
model = dict(
    type='CILRecognizer2D',
    backbone=dict(
        type='ResNetTSM',
        pretrained='https://download.pytorch.org/models/resnet50-0676ba61.pth',
        depth=50,
        norm_eval=False,
        num_segments=8,
        shift_div=8),
    cls_head=dict(
        type='IncrementalTSMHead',
        num_classes=starting_num_classes,
        in_channels=2048,
        inc_head_config=dict(type='LocalSimilarityClassifier',
                             out_features=starting_num_classes,
                             nb_proxies=1),
        num_segments=8,
        loss_cls=dict(type='LSCLoss'),
        spatial_type='avg',
        consensus=dict(type='AvgConsensus', dim=1),
        dropout_ratio=0.5,
        init_std=0.001,
        is_shift=True,
    ),
    # model training and testing settings
    train_cfg=None,
    test_cfg=dict(average_clips='prob'))

kd_modules_names = ['backbone.layer1', 'backbone.layer2', 'backbone.layer3', 'backbone.layer4', 'cls_head.avg_pool']
repr_hook = 'cls_head.avg_pool'  # extract representation
kd_exemplar_only = False
kd_weight_by_module = [0.5, 0.5, 0.5, 0.5, 1]
adaptive_scale_factors = [1.0, 4.219004621945797, 4.33589667773576, 4.449719092257398, 4.560701700396552,
                          4.669047011971501, 4.774934554525329, 4.878524367060187, 4.979959839195493,
                          5.079370039680118, 5.176871642217914, 5.272570530585627, 5.366563145999495,
                          5.458937625582473, 5.549774770204643, 5.639148871948674, 5.727128425310541,
                          5.813776741499453, 5.89915248150105]


# cil optimizer and lr_scheduler
optimizer = dict(
    type='SGD',
    constructor='CILTSMOptimizerConstructorImprovised',
    paramwise_cfg=dict(fc_lr_scale_factor=5.0),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=20, norm_type=2))
lr_scheduler = dict(type='MultiStepLR', params=dict(milestones=[20, 30], gamma=0.1))

# cbf optimizer and lr_scheduler
cbf_num_epochs_per_task = 50
cbf_optimizer = dict(
    type='SGD',
    constructor='CILTSMOptimizerConstructorImprovised',
    paramwise_cfg=dict(fc_lr_scale_factor=1.0),
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001)
cbf_lr_scheduler = dict(type='MultiStepLR', params=dict(milestones=[20, 30], gamma=0.1))

# dataset settings
data_root = os.path.join(data_dir, 'rawframes')
train_ann_file = os.path.join(data_dir, 'sthv2_train_list_rawframes.txt')
val_ann_file = os.path.join(data_dir, 'sthv2_val_list_rawframes.txt')
cil_ann_file_template = '{}_task_{}.txt'  # requre exact 2 placeholders

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)

train_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='RandAugment', n=2, m=10, prob=randAug_prob),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label', 'randAug'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    # dict(type='ThreeCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=8,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    # dict(type='ThreeCrop', crop_size=256),
    # dict(type='TenCrop', crop_size=256),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

# for extracting features to construct exemplar set
# this pipeline should be similar to validation pipeline if only run one epoch
# In case we ran multiple epochs, this pipeline should be similar to train pipeline
features_extraction_pipeline = [
    dict(type='SampleFrames', clip_len=1, frame_interval=1, num_clips=8, test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    # dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]

dataset_type = 'BackgroundMixDataset'
background_dir = os.path.join(data_dir, 'bg_extract')
data = dict(
    train=dict(
        type=dataset_type,
        ann_file='',  # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=train_pipeline,
        alpha=0.5,
        with_randAug=True
    ),
    val=dict(
        type=dataset_type,
        ann_file='',  # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=val_pipeline,
        test_mode=True),
    test=dict(
        type=dataset_type,
        ann_file='',  # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True),

    features_extraction=dict(
        type=dataset_type,
        ann_file='',  # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=features_extraction_pipeline,
        test_mode=True),
    features_extraction_epochs=1,  # this value should be set to 1 if there's no randomness in pipeline

    exemplar=dict(
        type=dataset_type,
        ann_file='',  # need to update this value before using
        bg_dir=background_dir,
        data_prefix=data_root,
        pipeline=train_pipeline),
)

keep_all_backgrounds = False
cbf_full_bg = False
