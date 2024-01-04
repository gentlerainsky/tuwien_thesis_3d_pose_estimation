_base_ = [
    # "mmpose::body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py"
    "mmpose::body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_udp-8xb32-210e_coco-384x288.py"
]

# data loaders
train_dataloader = dict(
    batch_size=16
)
val_dataloader = dict(
    batch_size=16
)
test_dataloader = dict(
    batch_size=16
)

# train_cfg = dict(_delete_=True, max_iters=1000)
visualizer = dict(vis_backends=[
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
    dict(type='WandbVisBackend'),
])
