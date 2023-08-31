from mmengine.config import Config
from mmengine.runner import Runner
from mmpose.datasets import CocoDataset
from mmpose.apis import init_model, MMPoseInferencer, inference_topdown
from mmpose.visualization import PoseLocalVisualizer
import mmcv
import matplotlib.pyplot as plt
import os

import numpy as np
from mmengine.structures import InstanceData
from mmpose.structures import PoseDataSample


class PoseEstimator2D:
    def __init__(
        self,
        config_path,
        pretrained_path,
        checkpoint_path,
        data_root_path,
        device='cpu',
        working_directory='./pose_estimator_2d_wd',
        log_level='INFO'
    ):
        self.config_path = config_path
        self.pretrained_path = pretrained_path
        self.checkpoint_path = checkpoint_path
        self.data_root = data_root_path
        self.device = device
        self.load_from_checkpoint = False
        self.working_directory = working_directory
        self.log_level = log_level
        # self.update_config()

    def update_config(self):
        self.config = Config.fromfile(self.config_path)
        self.config.data_root = self.data_root
        if self.load_from_checkpoint:
            self.config.load_from = self.checkpoint_path
        else:
            self.config.load_from = self.pretrained_path
        self.config.work_dir = self.working_directory
        self.config.log_level = self.log_level
        self.config.train_cfg['by_epoch'] = True
        # self.config.train_cfg['max_iters'] = 1000
        self.config.train_cfg['val_interval'] = 1
        self.config.train_cfg['max_epochs'] = 10

        self.config.train_dataloader['dataset']['data_root'] = self.data_root
        self.config.train_dataloader['dataset']['ann_file'] = os.path.join(
            self.config.data_root, 'annotations/person_keypoints_train.json')
        # self.config.train_dataloader['dataset']['ann_file'] = 'annotations/person_keypoints_train.json'
        self.config.val_dataloader['dataset']['bbox_file'] = os.path.join(
            self.config.data_root, 'person_detection_results/human_detection_train.json')
        self.config.train_dataloader['dataset']['data_prefix']['img'] = 'images/train'

        self.config.val_dataloader['dataset']['data_root'] = self.config.data_root
        self.config.val_dataloader['dataset']['ann_file'] = os.path.join(
            self.config.data_root, 'annotations/person_keypoints_val.json')
        # self.config.val_dataloader['dataset']['bbox_file'] = os.path.join(
        #   self.config.data_root, 'person_detection_results/ground_truth_human_detection_val.json')
        self.config.val_dataloader['dataset']['bbox_file'] = os.path.join(
            self.config.data_root, 'person_detection_results/human_detection_val.json')
        # self.config.val_dataloader['dataset']['ann_file'] = 'annotations/person_keypoints_val.json'
        # self.config.val_dataloader['dataset']['bbox_file'] = 'person_detection_results/ground_truth_val.json'
        self.config.val_dataloader['dataset']['data_prefix']['img'] = 'images/val'

        self.config.test_dataloader['dataset']['data_root'] = self.config.data_root
        self.config.test_dataloader['dataset']['ann_file'] = os.path.join(
            self.config.data_root, 'annotations/person_keypoints_test.json')
        # self.config.test_dataloader['dataset']['ann_file'] = 'annotations/person_keypoints_test.json'
        self.config.test_dataloader['dataset']['bbox_file'] = os.path.join(
            self.config.data_root, 'person_detection_results/ground_truth_human_detection_test.json')
        # self.config.test_dataloader['dataset']['bbox_file'] = 'person_detection_results/ground_truth_test.json'
        self.config.test_dataloader['dataset']['data_prefix']['img'] = 'images/test'
        self.config.val_evaluator.ann_file = os.path.join(self.config.data_root, 'annotations/person_keypoints_val.json')
        self.config.test_evaluator.ann_file = os.path.join(self.config.data_root, 'annotations/person_keypoints_test.json')
        # self.config.val_evaluator.ann_file = 'annotations/person_keypoints_val.json'
        # self.config.test_evaluator.ann_file = 'annotations/person_keypoints_test.json'
        self.runner = Runner.from_cfg(self.config)
        self.model = self.runner.model
        self.model.cfg = self.runner.cfg

    def load_pretrained(self):
        self.load_from_checkpoint = True
        self.update_config()

    def finetune(self):
        self.update_config()
        self.model.train()
        self.runner.train()
        with open(os.path.join(self.config.work_dir, 'last_checkpoint')) as f:
            filepath = f.readline()
            self.checkpoint_path = filepath

    def test(self):
        # self.update_config()
        self.runner.test()

    def inference(self, img_path, bboxes, bbox_format):
        img = mmcv.imread(img_path)
        result = inference_topdown(
            self.model,
            img,
            bboxes=bboxes,
            bbox_format=bbox_format)
        return result
    
    def get_2d_pose(self, img_path, bboxes, bbox_format):
        result = self.inference(img_path, bboxes, bbox_format)
        return result[0].pred_instances['keypoints'][0]

    def visualize(self, img_path, bboxes, bbox_format, gt_keypoints):
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        

        L = 0
        C = 1
        R = 2
        connections = [
            (0, 1, L, 'nose_left_eye'), # nose & left_eye
            (0, 2, R, 'nose_right_eye'), # nose & right_eye
            (1, 2, C, 'left_right_eye'), # left & right eyes
            (1, 3, L, 'left_ear_left_eye'), # left ear & eye
            (2, 4, R, 'right_ear_right_eye'), # right ear & eye
            # (0, 5, L, 'nose_left_shoulder'), # nose & left shoulder
            # (0, 6, R, 'nose_right_shoulder'), # nose & right shoulder
            # (3, 5, L, 'left_ear_shoulder'), # left ear & shoulder
            # (4, 6, R, 'right_ear_shoulder'), # right ear & shoulder
            (5, 6, C, 'left_shoulder_right_sholder'), # left & right shoulder
            (5, 7, L, 'left_sholder_left_elbow'), # left shoulder & elbow
            (5, 11, L, 'left_shoulder_left_hip'), # left shoulder & hip
            (6, 8, R, 'right_shoulder_right_elbow'), # right shoulder & elbow
            (6, 12, R, 'right_shoulder_right_hip'), # right shoulder & hip
            (7, 9, L, 'left_elbow_left_wrist'), # left elbow & wrist
            (8, 10, R, 'right_elbow_right_wrist'), # right elbow & wrist
            (11, 12, C, 'left_hip_right_hip'), # left & right hip
            (11, 13, L, 'left_hip_left_knee'), # left hip & knee
            (12, 14, R, 'right_hip_right_knee'), # right hip & knee
            (13, 15, L, 'left_knee_left_ankle'), # left knee & ankle
            (14, 16, R, 'right_knee_right_ankle') # right knee & ankle
        ]
        dataset_meta = {
            'skeleton_links': np.array(connections)[:, :2].astype(int).tolist()
        }
        pose_local_visualizer = PoseLocalVisualizer(
            vis_backends=[{'type': 'LocalVisBackend'}],
            radius=10,
            line_width=10,
            text_color=(255, 255, 0),
            backend='matplotlib',
            # kpt_color=(255, 255, 0),
            # det_kpt_color='green',
            # link_color='green'
            link_color=[
                (255, 0, 0),
                (255, 128, 0),
                (255, 255, 0),
                (128, 255, 0),
                (0, 255, 0),
                (0, 255, 128),
                (0, 255, 255),
                (0, 128, 255),
                (0, 0, 255),
                (64, 0, 255),
                (191, 0, 255),
                (255, 0, 64),
                (255, 0, 0),
                (140, 115, 115),
                (128, 50, 128),
                (0, 0, 0),
                (0, 0, 0),
            ]
        )
        gt_instances = InstanceData()
        gt_instances.keypoints = gt_keypoints
        data_sample = PoseDataSample()
        data_sample.gt_instances = gt_instances

        pose_local_visualizer.set_dataset_meta(dataset_meta=dataset_meta)
        
        pred_instances = InstanceData()
        inference_result = self.inference(img_path, bboxes, bbox_format)
        pred_instances.keypoints = inference_result[0].pred_instances['keypoints']
        pred_instances.score = inference_result[0].pred_instances['keypoint_scores']
        data_sample.pred_instances = pred_instances
        pose_local_visualizer.add_datasample(
            'detected',
            img,
            data_sample,
            show_kpt_idx=True,
        )

        imgplot = plt.imshow(pose_local_visualizer.get_image())
        plt.show()

if __name__ == '__main__':
    pass
