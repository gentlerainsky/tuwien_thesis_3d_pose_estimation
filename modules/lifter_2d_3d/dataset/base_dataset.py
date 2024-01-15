import json
import numpy as np
from modules.lifter_2d_3d.utils.normalization import (
    center_pose2d_to_neck,
    center_pose3d_to_neck,
    normalize_2d_pose_to_image,
    normalize_2d_pose_to_bbox,
    normalize_2d_pose_to_pose,
    normalize_rotation
)

class BaseDataset:
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
        is_normalize_rotation=None,
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
            raise ValueError(
                'is_normalize_to_pose and ' +
                'is_normalize_to_bbox cannot be both true.'
            )
        self.bbox_format = bbox_format
        self.is_normalize_rotation = is_normalize_rotation
        self.actors = set(actors)
        self.activities = set([])
        self.raw_data = []
        self.preprocess()

    def read_prediction_file(self):
        predictions = {}
        with open(self.prediction_file) as f:
            data = json.loads(f.read())
            for item in data:
                predictions[item["image_id"]] = item
        return predictions

    def read_bbox_file(self):
        bbox_info = {}
        with open(self.bbox_file) as f:
            data = json.loads(f.read())
            for item in data:
                bbox_info[item["image_id"]] = item
        return bbox_info

    def read_annotation_file(self):
        with open(self.annotation_file) as f:
            data = json.loads(f.read())
            metadata = data['categories'][0]
            camera_parameters = data['camera_parameters']
            images = [img for img in data["images"] if img["actor"] in self.actors]
            image_annotation_info = {item["id"]: item for item in data["annotations"]}
        return {
            'metadata': metadata,
            'camera_parameters': camera_parameters,
            'images': images,
            'image_annotation_info': image_annotation_info
        }

    def filter_relevance_joint(self, pose_2d, pose_3d):
        if self.exclude_ankle:
            pose_2d = pose_2d[:-2]
            pose_3d = pose_3d[:-2]
        if self.exclude_knee:
            pose_2d = pose_2d[:-2]
            pose_3d = pose_3d[:-2]
        # Drive & Act dataset specify unannotated joints
        # with a zero vector
        valid_keypoints = pose_3d.sum(axis=1) != 0
        return pose_3d, pose_3d, valid_keypoints

    def preprocess(self):
        predictions = self.read_prediction_file()
        bbox_info = self.read_bbox_file()
        annotation_info = self.read_annotation_file()

        self.metadata = annotation_info['metadata']
        self.camera_parameters = annotation_info['camera_parameters']
        images = annotation_info['images']
        image_ann_info = annotation_info['image_annotation_info']
        self.samples = []
        
        for idx, image_info in enumerate(images):
            ann_info = image_ann_info[image_info["id"]]
            if ann_info["id"] not in predictions:
                print(f'Annotation is not found for {ann_info["id"]}')
                continue
            
            # first 2 columns are x, y coordinate.
            # the last one is its confidence score.
            pose_2d = np.array(
                predictions[ann_info["id"]]["keypoints"]
            ).reshape(-1, 3)[:2]
            pose_3d = np.array(ann_info["keypoints3D"]).reshape(-1, 3)
            
            pose_2d, pose_3d, valid_kp = self.filter_relevance_joint(pose_2d, pose_3d)

            if not np.any(valid_kp):
                print(f'skipping problematic image {ann_info["id"]}')
                continue

            bbox = bbox_info[ann_info["id"]]['bbox']

            # default image root
            root_2d = np.array([0, 0])
            root_3d = np.array([0, 0, 0])
            # scale by the image resolution
            w = self.image_width
            h = self.image_height

            if self.is_normalize_to_bbox:
                keypoints2D, w, h = normalize_2d_pose_to_bbox(
                    keypoints2D, bbox, self.bbox_format
                )
            elif self.is_normalize_to_pose:
                keypoints2D, w, h = normalize_2d_pose_to_pose(keypoints2D)
            else:
                keypoints2D, w, h = normalize_2d_pose_to_image(
                    keypoints2D, self.image_width, self.image_height
                )
            if self.is_center_to_neck:
                keypoints2D, root_2d = center_pose2d_to_neck(keypoints2D)
                keypoints3D, root_3d = center_pose3d_to_neck(keypoints3D)
            if self.is_normalize_rotation:
                keypoints2D, keypoints3D = normalize_rotation(keypoints2D, keypoints3D)

            self.samples.append(
                {
                    "id": image_info["id"],
                    "array_idx": idx,
                    "filenames": image_info["file_name"],
                    "frame_id": image_info["frame_id"],
                    "actor": image_info["actor"],
                    "activity": image_info["activity"],
                    "keypoints2D": keypoints2D,
                    "keypoints3D": keypoints3D,
                    "valid": valid_kp,
                    "root_2d": root_2d,
                    "root_3d": root_3d,
                    "scale_factor": [w, h],
                    "bbox": bbox
                }
            )
            self.activities.add(image_info["activity"])

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
