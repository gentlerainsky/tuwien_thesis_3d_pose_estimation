import torch
import os
import pandas as pd
import json
import numpy as np


class DriveAndActKeypointDataset():
    def __init__(
            self,
            annotation_file,
            prediction_file,
            image_width,
            image_height,
            actors,
            exclude_ankle=False,
            exclude_hip=False
        ):
        self.annotation_file = annotation_file
        self.prediction_file = prediction_file
        self.image_width = image_width
        self.image_height = image_height
        self.exclude_ankle = exclude_ankle
        self.exclude_hip = exclude_hip
        self.actors = set(actors)
        self.activities = set([])
        self.raw_data = []
        self.init()

    def init(self):
        predictions = {}
        with open(self.prediction_file) as f:
            data = json.loads(f.read())
            for item in data:
                predictions[item['image_id']] = item
        with open(self.annotation_file) as f:
            data = json.loads(f.read())
            self.metadata = data['categories'][0]
            self.camera_parameters = data['camera_parameters']

            images = [img for img in data['images'] if img['actor'] in self.actors]
            annotations = {item['id']: item for item in data['annotations']}
            samples = []
            for idx, image in enumerate(images):
                annotation = annotations[image['id']]
                self.raw_data.append({
                    # 'keypoints2D': np.array(annotation['keypoints']).reshape(-1, 3),
                    'keypoints3D': np.array(annotation['keypoints3D']).reshape(-1, 3)
                })
                if annotation['id'] not in predictions:
                    print(f'skipping problematic image {annotation["id"]}')
                    continue
                keypoints2D = np.array(predictions[annotation['id']]['keypoints']).reshape(-1, 3)
                if self.exclude_hip:
                    keypoints2D = keypoints2D[:-4,:]
                elif self.exclude_ankle:
                    keypoints2D = keypoints2D[:-2,:]
                
                keypoints2D[:, 0] = keypoints2D[:, 0] / self.image_height
                keypoints2D[:, 1] = keypoints2D[:, 1] / self.image_width
                keypoints3D = np.array(annotation['keypoints3D']).reshape(-1, 3)
                if self.exclude_hip:
                    keypoints3D = keypoints3D[:-4,:]
                elif self.exclude_ankle:
                    keypoints3D = keypoints3D[:-2, :]
                valid_keypoints = (keypoints3D.sum(axis=1) != 0)
                samples.append({
                    'id': image['id'],
                    'array_idx': idx,
                    'filenames': image['file_name'],
                    'frame_id': image['frame_id'],
                    'actor': image['actor'],
                    'activity': image['activity'],
                    'keypoints2D': keypoints2D,
                    'keypoints3D': keypoints3D,
                    'valid': valid_keypoints
                })
                self.activities.add(image['activity'])
            self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx)-> dict:
        sample = self.samples[idx]
        return dict(
            img_id=sample['id'],
            arr_id=sample['array_idx'],
            keypoints_2d=sample['keypoints2D'][:, :2], 
            keypoints_3d=sample['keypoints3D'],
            valid=sample['valid'],
            activities=sample['activity']
        )
        # return sample

