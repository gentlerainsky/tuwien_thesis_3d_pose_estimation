import torch
import os
import pandas as pd
import json
import numpy as np


class SimpleKeypointDataset:
    def __init__(
        self,
        annotation_file,
        prediction_file,
        bbox_file,
        image_width,
        image_height,
        exclude_ankle=False,
        exclude_knee=False,
        is_normalize_to_bbox=False,
        bbox_format='xywh'
    ):
        self.annotation_file = annotation_file
        self.prediction_file = prediction_file
        self.bbox_file = bbox_file
        self.image_width = image_width
        self.image_height = image_height
        self.exclude_ankle = exclude_ankle
        self.exclude_knee = exclude_knee
        self.is_normalize_to_bbox = is_normalize_to_bbox
        self.bbox_format = bbox_format
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
            samples = []
            for idx, annotation in enumerate(data["annotations"]):
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

                bbox = bbox_info[annotation["id"]]['bbox']
                root_2d = np.array([0, 0])
                root_3d = np.array([0, 0, 0])
                if self.is_normalize_to_bbox:
                    # center
                    # use neck as the root. Neck is defined as the center of left/right shoulder
                    root_2d = (keypoints2D[5, :2] + keypoints2D[6, :2]) / 2
                    keypoints2D[:, :2] = keypoints2D[:, :2] - root_2d

                    root_3d = (keypoints3D[5, :] + keypoints3D[6, :]) / 2
                    keypoints3D = keypoints3D - root_3d

                    # scale by the bounding box
                    # note that 3D keypoints is usually already scaled.
                    x, y, w, h = bbox
                    if self.bbox_format == 'xyxy':
                        x, y, x2, y2 = bbox
                        w = x2 - x
                        h = y2 - y
                else:
                    # scale by the image resolution
                    w = self.image_width
                    h = self.image_height
                keypoints2D[:, 0] = keypoints2D[:, 0] / w
                keypoints2D[:, 1] = keypoints2D[:, 1] / h

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
                        "scale_factor": [w, h]
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
        # return sample
