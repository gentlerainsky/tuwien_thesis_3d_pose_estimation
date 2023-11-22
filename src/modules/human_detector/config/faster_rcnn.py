_base_ = [
    'mmdet::faster_rcnn/faster-rcnn_r50-caffe_fpn_ms-1x_coco-person.py'
]

# # # max_epochs
# train_cfg = dict(max_epochs=1)
# default_hooks = dict(logger=dict(interval=500))

# dataset_type = "CocoDataset"
# data_root = "/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/"

# train_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file="annotations/person_keypoints_train.json",
#         data_prefix=dict(img="images/train/"),
#         filter_cfg=dict(filter_empty_gt=True, min_size=32),
#     ),
# )

# val_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file="annotations/person_keypoints_val.json",
#         data_prefix=dict(img="images/val/"),
#         test_mode=True,
#     ),
# )

# test_dataloader = dict(
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file="annotations/person_keypoints_test.json",
#         data_prefix=dict(img="images/test/"),
#         test_mode=True,
#     ),
# )

# val_evaluator = dict(
#     ann_file=data_root + "annotations/person_keypoints_val.json",
# )
# test_evaluator = dict(
#     ann_file=data_root + "annotations/person_keypoints_test.json",
# )

# visualizer = dict(vis_backends=[
#     dict(type='LocalVisBackend'),
#     dict(type='TensorboardVisBackend'),
#     dict(type='WandbVisBackend'),
# ])

# work_dir = "human_detector_wd"