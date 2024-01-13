import torch
import os
import pandas as pd
import json
import numpy as np


class DriveAndActKeypointDataset:
    def __init__(
        self,
        annotation_file,
        prediction_file,
        bbox_file,
        image_width,
        image_height,
        actors,
        exclude_ankle=False,
        exclude_knee=False,
        is_center_to_neck=False,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=False,
        bbox_format='xywh'
    ):
        self.annotation_file = annotation_file
        self.prediction_file = prediction_file
        self.bbox_file = bbox_file
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
        self.actors = set(actors)
        self.activities = set([])
        self.raw_data = []
        self.init()

    def init(self):
        predictions = {}
        with open(self.prediction_file) as f:
            data = json.loads(f.read())
            for item in data:
                predictions[item["image_id"]] = item
        bbox_info = {}
        with open(self.bbox_file) as f:
            data = json.loads(f.read())
            for item in data:
                bbox_info[item["image_id"]] = item
        with open(self.annotation_file) as f:
            data = json.loads(f.read())
            self.metadata = data["categories"][0]
            self.camera_parameters = data["camera_parameters"]

            images = [img for img in data["images"] if img["actor"] in self.actors]
            annotations = {item["id"]: item for item in data["annotations"]}
            samples = []
            for idx, image in enumerate(images):
                annotation = annotations[image["id"]]
                self.raw_data.append(
                    {
                        # 'keypoints2D': np.array(annotation['keypoints']).reshape(-1, 3),
                        "keypoints3D": np.array(annotation["keypoints3D"]).reshape(
                            -1, 3
                        )
                    }
                )
                if annotation["id"] not in predictions:
                    print(f'skipping problematic image {annotation["id"]}')
                    continue
                keypoints2D = np.array(
                    predictions[annotation["id"]]["keypoints"]
                ).reshape(-1, 3)
                keypoints3D = np.array(annotation["keypoints3D"]).reshape(-1, 3)
                if self.exclude_ankle:
                    keypoints2D = keypoints2D[:-2, :]
                    keypoints3D = keypoints3D[:-2, :]
                if self.exclude_knee:
                    keypoints2D = keypoints2D[:-2, :]
                    keypoints3D = keypoints3D[:-2, :]
                valid_keypoints = keypoints3D.sum(axis=1) != 0
                bbox = bbox_info[annotation["id"]]['bbox']
                root_2d = np.array([0, 0])
                root_3d = np.array([0, 0, 0])

                # scale by the image resolution
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

                    root_3d = (keypoints3D[5, :] + keypoints3D[6, :]) / 2
                    keypoints3D = keypoints3D - root_3d
                keypoints2D[:, 0] = keypoints2D[:, 0] / w
                keypoints2D[:, 1] = keypoints2D[:, 1] / h
                
                if (not np.any(valid_keypoints)):
                    print(f'skipping problematic image {annotation["id"]}')
                    continue
                samples.append(
                    {
                        "id": image["id"],
                        "array_idx": idx,
                        "filenames": image["file_name"],
                        "frame_id": image["frame_id"],
                        "actor": image["actor"],
                        "activity": image["activity"],
                        "keypoints2D": keypoints2D,
                        "keypoints3D": keypoints3D,
                        "valid": valid_keypoints,
                        "root_2d": root_2d,
                        "root_3d": root_3d,
                        "scale_factor": [w, h],
                        "bbox": bbox
                    }
                )
                self.activities.add(image["activity"])
            self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        sample = self.samples[idx]
        return dict(
            img_id=sample["id"],
            arr_id=sample["array_idx"],
            keypoints_2d=sample["keypoints2D"][:, :2],
            keypoints_3d=sample["keypoints3D"],
            valid=sample["valid"],
            activities=sample["activity"],
        )
        # return sample
