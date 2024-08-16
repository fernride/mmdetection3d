# model settings
_base_ = [
    '../_base_/datasets/fern3d.py',
    '../_base_/default_runtime.py'
]

#point_cloud_range = [ 0, -39.68, -1, 69.12, 39.68, 3]
point_cloud_range = [-20.0, -39.68, -0.25, 49.12, 39.68, 3.75]
x_min = point_cloud_range[0] + 1.0
x_max = point_cloud_range[3] - 1.0
y_min = point_cloud_range[1] + 1.0
y_max = point_cloud_range[4] - 1.0

anchors_info = {
    "truck": {
        "ranges": [x_min, y_min, -1.0, x_max, y_max, 0.5],
        "sizes": [5.94, 2.65, 3.65],
    },
    "trailer": {
        "ranges": [x_min, y_min, -1.0, x_max, y_max, 0.5],
        "sizes": [12.8, 2.75, 3.38],
    },
    "human": {
        "ranges": [max(x_min, -12.0), max(y_min, -20.0), -0.5, min(x_max, 35.0), min(y_max, 20.0), 1.5],
        "sizes": [0.8, 0.7, 1.78],
    },
    "car": {
        "ranges": [max(x_min, 0.), max(y_min, -20.0), -1.0, min(x_max, 50.0), min(y_max, 20.0), 0.5],
        "sizes": [4.43, 1.77, 1.70],
    },
    "crane": {
        "ranges": [max(x_min, 0.), max(y_min, -20.0), -1.0, min(x_max, 50.0), min(y_max, 20.0), 0.5],
        "sizes": [10.5, 1.0, 2.2],
    },
    "forklift": {
        "ranges": [max(x_min, 0.), max(y_min, -20.0), -1.0, min(x_max, 40.0), min(y_max, 20.0), 0.5],
        "sizes": [2.5, 1.2, 1.93],
    },
    "reach_stacker": {
        "ranges": [max(x_min, 0.), max(y_min, -20.0), -1.0, min(x_max, 50.0), min(y_max, 20.0), 0.5],
        "sizes": [6.0, 2.5, 2.5],
    },
}

bbox_assigner = {
    "truck": dict(
        type='Max3DIoUAssigner',
        iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
        pos_iou_thr=0.6,
        neg_iou_thr=0.45,
        min_pos_iou=0.45,
        ignore_iof_thr=-1),
    "trailer": dict(
        type='Max3DIoUAssigner',
        iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
        pos_iou_thr=0.6,
        neg_iou_thr=0.45,
        min_pos_iou=0.45,
        ignore_iof_thr=-1),
    "human": dict(
        type='Max3DIoUAssigner',
        iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
        pos_iou_thr=0.4,
        neg_iou_thr=0.3,
        min_pos_iou=0.3,
        ignore_iof_thr=-1),
    "car": dict(
        type='Max3DIoUAssigner',
        iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
        pos_iou_thr=0.6,
        neg_iou_thr=0.45,
        min_pos_iou=0.45,
        ignore_iof_thr=-1),
    "crane": dict(
        type='Max3DIoUAssigner',
        iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
        pos_iou_thr=0.6,
        neg_iou_thr=0.45,
        min_pos_iou=0.45,
        ignore_iof_thr=-1),
    "forklift": dict(
        type='Max3DIoUAssigner',
        iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
        pos_iou_thr=0.6,
        neg_iou_thr=0.45,
        min_pos_iou=0.45,
        ignore_iof_thr=-1),
    "reach_stacker": dict(
        type='Max3DIoUAssigner',
        iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
        pos_iou_thr=0.6,
        neg_iou_thr=0.45,
        min_pos_iou=0.45,
        ignore_iof_thr=-1),
}

class_names = ['car', 'truck', 'trailer', 'human', 'reach_stacker']
#class_names = ['human']
# todo figure-out order of anchors vs order of classes in meta of pickle vs order of classes in config
# from KITTI it looks like order in the config is the main

# TODO define ranges
voxel_size = [0.16, 0.16, 4]
scatter_shape = [496, 432] # y x

model = dict(
    type='VoxelNet',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=20,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(32000, 32000))),
    voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range),
    middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=scatter_shape),
    backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=len(class_names),
        in_channels=384,
        feat_channels=384,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[anchors_info[class_name]['ranges'] for class_name in class_names],
            sizes=[anchors_info[class_name]['sizes'] for class_name in class_names],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[bbox_assigner[class_name] for class_name in class_names],
        allowed_border=0,
        pos_weight=-1,
        debug=True),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50))

work_dir = '/home/omuratov/workspace/fern_ml/output/fernnet'

default_hooks = dict(
    checkpoint=dict(
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        save_optimizer=True,
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='Det3DVisualizationHook', draw=False, vis_task="lidar_det", test_out_dir=f'{work_dir}/test_out_dir'),)

vis_backends = [dict(type='LocalVisBackend'), dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')


lr = 0.001

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01),
    clip_grad=dict(max_norm=10, norm_type=2))

coarse_optimization_iter = [0, 40]
fine_optimization_iter = [40, 2*1500]


param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=coarse_optimization_iter[1]-coarse_optimization_iter[0],
        eta_min=lr * 10,
        begin=coarse_optimization_iter[0],
        end=coarse_optimization_iter[1],
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=fine_optimization_iter[1]-fine_optimization_iter[0],
        eta_min=lr * 1e-4,
        begin=fine_optimization_iter[0],
        end=fine_optimization_iter[1],
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    dict(
        type='CosineAnnealingMomentum',
        T_max=coarse_optimization_iter[1]-coarse_optimization_iter[0],
        eta_min=0.85 / 0.95,
        begin=coarse_optimization_iter[0],
        end=coarse_optimization_iter[1],
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=fine_optimization_iter[1]-fine_optimization_iter[0],
        eta_min=1,
        begin=fine_optimization_iter[0],
        end=fine_optimization_iter[1],
        by_epoch=True,
        convert_to_iter_based=True)
]

train_cfg = dict(by_epoch=True, max_epochs=fine_optimization_iter[1], val_interval=1)
val_cfg = dict()
test_cfg = dict()
auto_scale_lr = dict(enable=False, base_batch_size=48)