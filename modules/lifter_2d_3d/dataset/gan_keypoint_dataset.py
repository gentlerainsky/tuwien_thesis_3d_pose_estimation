import json
import numpy as np
from modules.lifter_2d_3d.utils.normalization import (
    center_pose2d_to_neck,
    center_pose3d_to_neck,
    normalize_2d_pose_to_image,
    normalize_2d_pose_to_bbox,
    normalize_2d_pose_to_pose,
    normalize_rotation,
    rotate2D_to_x_axis
)

class GANKeypointDataset:
    def __init__(
        self,
        pose_2d,
        pose_3d,
        actors=None,
        exclude_ankle=True,
        exclude_knee=True,
        is_silence=True,
        is_center_to_neck=False,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=False,
        is_normalize_rotation=None,
        bbox_format='xywh',
        remove_activities=None,
        is_gt_2d_pose=False,
        subset_percentage=100
    ):
        self.input_pose_2d = pose_2d
        self.input_pose_3d = pose_3d
        self.exclude_ankle = exclude_ankle
        self.exclude_knee = exclude_knee
        self.exclude_ankle = exclude_ankle
        self.exclude_knee = exclude_knee
        self.is_center_to_neck = is_center_to_neck
        self.is_silence = is_silence
        self.is_normalize_to_bbox = is_normalize_to_bbox
        self.is_normalize_to_pose = is_normalize_to_pose
        self.is_gt_2d_pose = is_gt_2d_pose
        if (is_normalize_to_pose and is_normalize_to_bbox):
            raise ValueError(
                'is_normalize_to_pose and ' +
                'is_normalize_to_bbox cannot be both true.'
            )
        self.bbox_format = bbox_format
        self.is_normalize_rotation = is_normalize_rotation
        self.actors = actors
        self.remove_activities = remove_activities
        subset_percentage /= 100
        self.subset_percentage = subset_percentage
        if remove_activities is None:
            self.remove_activities = []
        self.pose_2d = []
        self.all_pose_3d = []
        self.pose_3d = []
        self.pose_3d_valid = []
        self.preprocess_2d()
        self.preprocess_3d()

    def preprocess_2d(self):
        for pose_2d in self.input_pose_2d:
            pose_2d = np.copy(pose_2d)
            if self.exclude_ankle:
                pose_2d = pose_2d[:-2]
            if self.exclude_knee:
                pose_2d = pose_2d[:-2]
            # Drive & Act dataset specify unannotated joints
            # with a zero vector
            root_2d = np.array([0, 0])
            if self.is_center_to_neck:
                pose_2d, root_2d = center_pose2d_to_neck(pose_2d)
            if self.is_normalize_to_pose:
                pose_2d, w, h = normalize_2d_pose_to_pose(pose_2d)
            if self.is_normalize_rotation:
                # pose_2d, pose_3d = normalize_rotation(pose_2d, pose_3d)
                left_shoulder_index = 5
                right_shoulder_index = 6
                pose_2d[:, :2], rotation_matrix = rotate2D_to_x_axis(
                    pose_2d[left_shoulder_index: right_shoulder_index + 1, :2],
                    pose_2d[:, :2]
                )
            self.pose_2d.append(pose_2d)

    def preprocess_3d(self):
        for pose_3d in self.input_pose_3d:
            pose_3d = np.copy(pose_3d)
            if self.exclude_ankle:
                pose_3d = pose_3d[:-2]
            if self.exclude_knee:
                pose_3d = pose_3d[:-2]
            # Drive & Act dataset specify unannotated joints
            # with a zero vector
            valid_kp = (pose_3d.sum(axis=1) != 0)
            if not np.any(valid_kp):
                continue
            # left_shoulder_index=5, right_shoulder_index=6
            root_3d = np.array([0, 0, 0])
            # scale by the image resolution
            if self.is_center_to_neck:
                pose_3d, root_3d = center_pose3d_to_neck(pose_3d)
            self.all_pose_3d.append(dict(
                pose_3d=pose_3d,
                valid=valid_kp
            ))
        self.shuffle()

    def shuffle(self):        
        self.pose_3d = np.random.choice(self.all_pose_3d, int(self.subset_percentage * len(self.all_pose_3d)))

    def __len__(self):
        return len(self.pose_3d)
    
    def __getitem__(self, idx) -> dict:
        pose_2d = self.pose_2d[idx]
        pose_3d_ann = self.pose_3d[idx]
        pose_3d = pose_3d_ann['pose_3d']
        valid = pose_3d_ann['valid']
        item = dict(
            keypoints_2d=pose_2d[:, :2].astype(np.float32),
            keypoints_3d=pose_3d.astype(np.float32),
            valid=valid
        )
        return item
