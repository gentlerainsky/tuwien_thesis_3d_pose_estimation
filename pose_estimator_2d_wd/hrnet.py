default_scope = 'mmpose'
default_hooks = dict(
    timer=dict(type='IterTimerHook', _scope_='mmpose'),
    logger=dict(type='LoggerHook', interval=50, _scope_='mmpose'),
    param_scheduler=dict(type='ParamSchedulerHook', _scope_='mmpose'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='coco/AP',
        rule='greater',
        _scope_='mmpose'),
    sampler_seed=dict(type='DistSamplerSeedHook', _scope_='mmpose'),
    visualization=dict(
        type='PoseVisualizationHook', enable=False, _scope_='mmpose'))
custom_hooks = [
    dict(type='SyncBuffersHook', _scope_='mmpose'),
]
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [
    dict(type='LocalVisBackend', _scope_='mmpose'),
]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ],
    name='visualizer',
    _scope_='mmpose')
log_processor = dict(
    type='LogProcessor',
    window_size=50,
    by_epoch=True,
    num_digits=6,
    _scope_='mmpose')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth'
resume = False
backend_args = dict(backend='local')
train_cfg = dict(max_iters=1000, by_epoch=False, val_interval=100)
val_cfg = dict()
test_cfg = dict()
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.0005, _scope_='mmpose'))
param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=500,
        start_factor=0.001,
        by_epoch=False,
        _scope_='mmpose'),
    dict(
        type='MultiStepLR',
        begin=0,
        end=210,
        milestones=[
            170,
            200,
        ],
        gamma=0.1,
        by_epoch=True,
        _scope_='mmpose'),
]
auto_scale_lr = dict(base_batch_size=512)
codec = dict(
    type='MSRAHeatmap',
    input_size=(
        192,
        256,
    ),
    heatmap_size=(
        48,
        64,
    ),
    sigma=2,
    _scope_='mmpose')
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        bgr_to_rgb=True),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4, ),
                num_channels=(64, )),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                ),
                num_channels=(
                    32,
                    64,
                )),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                ),
                num_channels=(
                    32,
                    64,
                    128,
                )),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(
                    4,
                    4,
                    4,
                    4,
                ),
                num_channels=(
                    32,
                    64,
                    128,
                    256,
                ))),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=
            'https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth'
        )),
    head=dict(
        type='HeatmapHead',
        in_channels=32,
        out_channels=17,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=dict(
            type='MSRAHeatmap',
            input_size=(
                192,
                256,
            ),
            heatmap_size=(
                48,
                64,
            ),
            sigma=2)),
    test_cfg=dict(flip_test=True, flip_mode='heatmap', shift_heatmap=True),
    _scope_='mmpose')
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/coco/'
train_pipeline = [
    dict(type='LoadImage', _scope_='mmpose'),
    dict(type='GetBBoxCenterScale', _scope_='mmpose'),
    dict(type='RandomFlip', direction='horizontal', _scope_='mmpose'),
    dict(type='RandomHalfBody', _scope_='mmpose'),
    dict(type='RandomBBoxTransform', _scope_='mmpose'),
    dict(type='TopdownAffine', input_size=(
        192,
        256,
    ), _scope_='mmpose'),
    dict(
        type='GenerateTarget',
        encoder=dict(
            type='MSRAHeatmap',
            input_size=(
                192,
                256,
            ),
            heatmap_size=(
                48,
                64,
            ),
            sigma=2),
        _scope_='mmpose'),
    dict(type='PackPoseInputs', _scope_='mmpose'),
]
val_pipeline = [
    dict(type='LoadImage', _scope_='mmpose'),
    dict(type='GetBBoxCenterScale', _scope_='mmpose'),
    dict(type='TopdownAffine', input_size=(
        192,
        256,
    ), _scope_='mmpose'),
    dict(type='PackPoseInputs', _scope_='mmpose'),
]
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True, _scope_='mmpose'),
    dataset=dict(
        type='CocoDataset',
        data_root='/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/',
        data_mode='topdown',
        ann_file='annotations/person_keypoints_train.json',
        data_prefix=dict(img='images/train'),
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='RandomFlip', direction='horizontal'),
            dict(type='RandomHalfBody'),
            dict(type='RandomBBoxTransform'),
            dict(type='TopdownAffine', input_size=(
                192,
                256,
            )),
            dict(
                type='GenerateTarget',
                encoder=dict(
                    type='MSRAHeatmap',
                    input_size=(
                        192,
                        256,
                    ),
                    heatmap_size=(
                        48,
                        64,
                    ),
                    sigma=2)),
            dict(type='PackPoseInputs'),
        ],
        _scope_='mmpose'))
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(
        type='DefaultSampler', shuffle=False, round_up=False,
        _scope_='mmpose'),
    dataset=dict(
        type='CocoDataset',
        data_root='/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/',
        data_mode='topdown',
        ann_file=
        '/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/annotations/person_keypoints_val.json',
        bbox_file=
        '/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/person_detection_results/ground_truth_val.json',
        data_prefix=dict(img='images/val'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(
                192,
                256,
            )),
            dict(type='PackPoseInputs'),
        ],
        _scope_='mmpose'))
test_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(
        type='DefaultSampler', shuffle=False, round_up=False,
        _scope_='mmpose'),
    dataset=dict(
        type='CocoDataset',
        data_root='/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/',
        data_mode='topdown',
        ann_file=
        '/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/annotations/person_keypoints_test.json',
        bbox_file=
        '/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/person_detection_results/ground_truth_test.json',
        data_prefix=dict(img='images/test'),
        test_mode=True,
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=(
                192,
                256,
            )),
            dict(type='PackPoseInputs'),
        ],
        _scope_='mmpose'))
val_evaluator = dict(
    type='CocoMetric',
    ann_file=
    '/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/annotations/person_keypoints_val.json',
    _scope_='mmpose')
test_evaluator = dict(
    type='CocoMetric',
    ann_file=
    '/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/annotations/person_keypoints_test.json',
    _scope_='mmpose')
work_dir = './pose_estimator_2d_wd'
