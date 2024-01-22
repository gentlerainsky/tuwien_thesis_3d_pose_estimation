import torch
import os
import pandas as pd
import json
import numpy as np
from modules.lifter_2d_3d.utils.normalization import rotate2D_to_x_axis, rotate3D_to_x_axis


class GroundTruthKeypointDataset:
    def __init__(
        self,
        annotation_file,
        image_width,
        image_height,
        exclude_ankle=False,
        exclude_knee=False,
        is_center_to_neck=False,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=False,
        is_normalize_rotation=None,
        bbox_format='xywh',
        remove_activities=[]
    ):
        self.annotation_file = annotation_file
        self.image_width = image_width
        self.image_height = image_height
        self.exclude_ankle = exclude_ankle
        self.exclude_knee = exclude_knee
        self.is_center_to_neck = is_center_to_neck
        self.is_normalize_to_bbox = is_normalize_to_bbox
        self.is_normalize_to_pose = is_normalize_to_pose
        if (is_normalize_to_pose and is_normalize_to_bbox):
            raise ValueError(f'is_normalize_to_pose and is_normalize_to_bbox cannot be both true.')
        self.bbox_format = bbox_format
        self.is_normalize_rotation = is_normalize_rotation
        self.activities = set([])
        self.image_activities = []
        self.remove_activities = remove_activities
        self.raw_data = []
        self.init()

    def init(self):
        with open(self.annotation_file) as f:
            data = json.loads(f.read())
            self.metadata = data["categories"][0]
            self.camera_parameters = data["camera_parameters"]
            samples = []
            for idx, annotation in enumerate(data["annotations"]):
                self.raw_data.append(
                    {
                        "keypoints2D": np.array(annotation["keypoints"]).reshape(-1, 3),
                        "keypoints3D": np.array(annotation["keypoints3D"]).reshape(
                            -1, 3
                        ),
                    }
                )
                keypoints2D = np.array(annotation["keypoints"]).reshape(-1, 3)
                keypoints3D = np.array(annotation["keypoints3D"]).reshape(-1, 3)
                bbox = annotation['bbox']
                bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[0] + bbox[2],
                    bbox[1] + bbox[3],
                ]
                if self.exclude_ankle:
                    keypoints2D = keypoints2D[:-2, :]
                    keypoints3D = keypoints3D[:-2, :]
                if self.exclude_knee:
                    keypoints2D = keypoints2D[:-2, :]
                    keypoints3D = keypoints3D[:-2, :]

                bbox = annotation["bbox"]
                root_2d = np.array([0, 0])
                root_3d = np.array([0, 0, 0])
                w = self.image_width
                h = self.image_height
                if self.is_normalize_to_bbox:
                    # scale by the bounding box
                    # note that 3D keypoints is usually already scaled.
                    x, y, w, h = bbox
                    if self.bbox_format == 'xyxy':
                        x, y, x2, y2 = bbox
                        w = x2 - x
                        h = y2 - y
                if self.is_normalize_to_pose:
                    # scale by the max-min position of 2D poses
                    x_max, y_max = np.max(keypoints2D[:, :2], axis=0)
                    x_min, y_min = np.min(keypoints2D[:, :2], axis=0)
                    # max_length = np.max([x_max - x_min, y_max - y_min])
                    w = x_max - x_min
                    h = y_max - y_min
                    # w = max_length
                    # h = max_length
                    bbox = [x_min, y_min, x_max, y_max]
                
                if self.is_center_to_neck:
                    # center
                    # use neck as the root. Neck is defined as the center of left/right shoulder
                    root_2d = (keypoints2D[5, :2] + keypoints2D[6, :2]) / 2
                    keypoints2D[:, :2] = keypoints2D[:, :2] - root_2d

                    root_3d = (keypoints3D[5] + keypoints3D[6]) / 2
                    keypoints3D = keypoints3D - root_3d

                if self.is_normalize_rotation:
                    # x, y = keypoints2D[5, 0], keypoints2D[5, 1]
                    # rad = np.arctan2(y, x)
                    keypoints2D[:, :2], _ = rotate2D_to_x_axis(keypoints2D[5:7, :2], keypoints2D[:, :2])
                    keypoints3D, _, _ = rotate3D_to_x_axis(keypoints3D[5:7], keypoints3D)

                valid_keypoints = keypoints3D.sum(axis=1) != 0
                samples.append(
                    {
                        "id": annotation["id"],
                        "filenames": data["images"][idx]["file_name"],
                        "keypoints2D": keypoints2D,
                        "keypoints3D": keypoints3D,
                        "valid": valid_keypoints,
                        "root_2d": root_2d,
                        "root_3d": root_3d,
                        "scale_factor": [w, h],
                        "bbox": bbox
                    }
                )
            self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        sample = self.samples[idx]
        return dict(
            img_id=sample["id"],
            keypoints_2d=sample["keypoints2D"][:, :2],
            keypoints_3d=sample["keypoints3D"],
            valid=sample["valid"],
        )
