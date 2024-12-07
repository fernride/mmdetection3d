# dataset description
dataset_type = 'Fern3dDataset'
#data_root = '/home/omuratov/bigdata/datasets/fern3d_v0_tiny/'
dataset_folder = '/home/omuratov/bigdata/datasets' 
#data_root = f'{dataset_folder}/fern3d_b0_b3_filtered/'
#data_root = f'{dataset_folder}/fern3d_combined'
#data_root = f'{dataset_folder}/fern_v0_v3_v4_v7_filtered'
#data_root = f'{dataset_folder}/fern3d_static_v0'
data_root = f'{dataset_folder}/fern3d_b1-b7_all'
#data_root = '/home/omuratov/bigdata/pipeline_v0/segments/00000/training/scan'
#class_names = ['car', 'truck', 'trailer', 'human', 'reach_stacker', 'crane', 'forklift']
class_names = [
    "car",
    "truck",
    "trailer",
    "human",
    "reach_stacker",
    "crane",
    "forklift",
    "mast",
    "barrier",
    "sign",
    "machine_other",]
#class_names = ["mast", "barrier", "sign"]
#point_cloud_range = [ 0, -39.68, -1, 50.00, 39.68, 3]
point_cloud_range = [-20.0, -51.20, -0.25, 69.6, 51.20, 19.75]
input_modality = dict(use_lidar=True, use_camera=False)
metainfo = dict(classes=class_names)
#metainfo = dict(classes=['human'])#class_names)

samples_folder = f'{dataset_folder}/fern_samples/'

backend_args = None
deg_to_rad_mult=3.14159265358979323846 / 180.0


db_sampler = dict(
    backend_args=None,
    classes=[
        'human',
    ],
    data_root=samples_folder,
    info_path=f'{samples_folder}/samples_train.pkl',
    points_loader=dict(
        backend_args=None,
        coord_type='LIDAR',
        load_dim=4,
        type='LoadPointsFromFile',
        use_dim=4),
    prepare=dict(
        filter_by_difficulty=[
            -1,
        ],
        filter_by_min_points=dict(human=50)),
    rate=1.0,
    sample_groups=dict(human=5))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    #dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomJitterPoints',
        jitter_std=[0.05, 0.05, 0.1],
        clip_range=[0.1, 0.1, 0.1],
    ),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5, flip_box3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-deg_to_rad_mult*10.0, deg_to_rad_mult*10.0],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.15, 0.15, 0.15]),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
#    dict(type='ObjectNameFilter', classes=['mast']),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            #dict(
            #    type='GlobalRotScaleTrans',
            #    rot_range=[0, 0],
            #    scale_ratio_range=[1., 1.],
            #    translation_std=[0, 0, 0]),
            #dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range)
        ]),
    #dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    #dict(type='ObjectNameFilter', classes=['mast']),
    dict(type='Pack3DDetInputs', keys=['points'])
]

eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        backend_args=backend_args),
    dict(type='Pack3DDetInputs', keys=['points'])
]

data_prefix = dict(pts='points/', img='', sweeps='')
batch_mult = 8
batch_mult = 6
train_dataloader = dict(
    batch_size=batch_mult*6,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='fern3d_train.pkl',
            data_prefix=data_prefix,
            pipeline=train_pipeline,
            modality=input_modality,
            test_mode=False,
            metainfo=metainfo,
            box_type_3d='LiDAR',
            filter_empty_gt=True,
            backend_args=backend_args)))

test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='fern3d_test.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        data_prefix=data_prefix,
        test_mode=True,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='fern3d_test.pkl',
        pipeline=test_pipeline,
        metainfo=metainfo,
        modality=input_modality,
        test_mode=True,
        data_prefix=data_prefix,
        box_type_3d='LiDAR',
        backend_args=backend_args))

val_evaluator = dict(
    type='FernDynamicMetric',
    ann_file=data_root + 'fern3d_test.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
