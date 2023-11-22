import torch
import os
import pandas as pd
import json
import numpy as np


class GroundTruthKeypointDataset():
    def __init__(
            self,
            annotation_file,
            image_width,
            image_height,
            exclude_ankle=False
        ):
        self.annotation_file = annotation_file
        self.image_width = image_width
        self.image_height = image_height
        self.exclude_ankle = exclude_ankle
        self.raw_data = []
        self.init()

    def init(self):
        with open(self.annotation_file) as f:
            data = json.loads(f.read())
            self.metadata = data['categories'][0]
            self.camera_parameters = data['camera_parameters']
            samples = []
            for idx, annotation in enumerate(data['annotations']):
                self.raw_data.append({
                    'keypoints2D': np.array(annotation['keypoints']).reshape(-1, 3),
                    'keypoints3D': np.array(annotation['keypoints3D']).reshape(-1, 3)
                })
                keypoints2D = np.array(annotation['keypoints']).reshape(-1, 3)
                if self.exclude_ankle:
                    keypoints2D = keypoints2D[:-2,:]
                keypoints2D[:, 0] = keypoints2D[:, 0] / self.image_height
                keypoints2D[:, 1] = keypoints2D[:, 1] / self.image_width
                keypoints3D = np.array(annotation['keypoints3D']).reshape(-1, 3)
                if self.exclude_ankle:
                    keypoints3D = keypoints3D[:-2, :]
                samples.append({
                    'filenames': data['images'][idx]['file_name'],
                    'keypoints2D': keypoints2D,
                    'keypoints3D': keypoints3D
                })
            self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        sample = self.samples[idx]
        return dict(
            keypoints_2d=sample['keypoints2D'][:, :2],
            keypoints_3d=sample['keypoints3D']
        )

