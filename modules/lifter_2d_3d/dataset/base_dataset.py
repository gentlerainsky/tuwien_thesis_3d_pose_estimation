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

class BaseDataset:
    def __init__(
        self,
        annotation_file,
        prediction_file,
        bbox_file,
        image_width,
        image_height,
        actors=None,
        exclude_ankle=False,
        exclude_knee=False,
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
        self.annotation_file = annotation_file
        self.prediction_file = prediction_file
        self.bbox_file = bbox_file
        self.image_width = image_width
        self.image_height = image_height
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
        if actors is not None:
            self.actors = set(self.actors)
        self.activities = set([])
        self.raw_data = []
        self.remove_activities = remove_activities
        if remove_activities is None:
            self.remove_activities = []
        self.image_activities = []
        self.activity_image_map = {}
        subset_percentage /= 100
        self.subset_percentage = subset_percentage
        self.sample_weight = None
        self.preprocess()

    def read_prediction_file(self):
        predictions = {}
        with open(self.prediction_file) as f:
            data = json.loads(f.read())
            if self.is_gt_2d_pose:
                data = data['annotations']
            for item in data:
                predictions[item['image_id']] = item
        return predictions

    def read_bbox_file(self):
        bbox_info = {}
        with open(self.bbox_file) as f:
            data = json.loads(f.read())
            for item in data:
                bbox_info[item['image_id']] = item
        return bbox_info

    def read_annotation_file(self):
        with open(self.annotation_file) as f:
            data = json.loads(f.read())
            metadata = data['categories'][0]
            camera_parameters = data['camera_parameters']
            images = data['images']
            if self.actors is not None:
                images = [img for img in data['images'] if img['actor'] in self.actors]
            image_annotation_info = {item['id']: item for item in data['annotations']}
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
        valid_keypoints = (pose_3d.sum(axis=1) != 0)
        return pose_2d, pose_3d, valid_keypoints

    def filter_samples(self, images):
        if len(self.remove_activities) > 0:
            return filter(
                lambda x: x['activity'] not in self.remove_activities,
                images
            )
        return images

    def preprocess(self):
        predictions = self.read_prediction_file()
        bbox_info = self.read_bbox_file()
        annotation_info = self.read_annotation_file()

        self.metadata = annotation_info['metadata']
        self.camera_parameters = annotation_info['camera_parameters']
        images = annotation_info['images']
        image_ann_info = annotation_info['image_annotation_info']
        self.samples = []
        images = self.filter_samples(images)
        for idx, image_info in enumerate(images):
            ann_info = image_ann_info[image_info['id']]
            if ann_info['id'] not in predictions:
                if not self.is_silence:
                    print(f'Annotation is not found for {ann_info["id"]}')
                continue
            # first 2 columns are x, y coordinate.
            # the last one is its confidence score.
            pose_2d = np.array(
                predictions[ann_info['id']]['keypoints']
            ).reshape(-1, 3)[:, :2]
            pose_3d = np.array(ann_info['keypoints3D']).reshape(-1, 3)
            pose_2d, pose_3d, valid_kp = self.filter_relevance_joint(pose_2d, pose_3d)
            if not np.any(valid_kp):
                if not self.is_silence:
                    print(f'skipping problematic image {ann_info["id"]}')
                continue

            # left_shoulder_index=5, right_shoulder_index=6
            # if (pose_3d[5].sum() == 0) or (pose_3d[6].sum() == 0):
            #     if not self.is_silence:
            #         print(f'skip images which both shoulders are not visible. {ann_info["id"]}')
            #     continue

            # if (pose_3d[valid_kp].shape[0] < (pose_3d.shape[0] // 4)):
            if (pose_3d[valid_kp].shape[0] < 5):
                if not self.is_silence:
                    print(f'skip images which contains too few visible keypoints. {ann_info["id"]}')
                continue
            bbox = bbox_info[ann_info['id']]['bbox']

            # default image root
            root_2d = np.array([0, 0])
            root_3d = np.array([0, 0, 0])
            # scale by the image resolution
            w = self.image_width
            h = self.image_height
            raw_pose_2d = np.copy(pose_2d)
            if self.is_center_to_neck:
                pose_2d, root_2d = center_pose2d_to_neck(pose_2d)
                pose_3d, root_3d = center_pose3d_to_neck(pose_3d)
            if self.is_normalize_to_bbox:
                pose_2d, w, h = normalize_2d_pose_to_bbox(
                    pose_2d, bbox, self.bbox_format
                )
            elif self.is_normalize_to_pose:
                pose_2d, w, h = normalize_2d_pose_to_pose(pose_2d)
            else:
                pose_2d, w, h = normalize_2d_pose_to_image(
                    pose_2d, self.image_width, self.image_height
                )

            if self.is_normalize_rotation:
                # pose_2d, pose_3d = normalize_rotation(pose_2d, pose_3d)
                pose_2d = np.copy(pose_2d)
                left_shoulder_index = 5
                right_shoulder_index = 6
                pose_2d[:, :2], rotation_matrix = rotate2D_to_x_axis(
                    pose_2d[left_shoulder_index: right_shoulder_index + 1, :2],
                    pose_2d[:, :2]
                )

            item = {
                'id': image_info['id'],
                'filenames': image_info['file_name'],
                'frame_id': image_info.get('frame_id', None),
                'actor': image_info.get('actor', None),
                'activity': image_info.get('activity', None),
                'raw_pose_2d': raw_pose_2d,
                'pose_2d': pose_2d,
                'pose_3d': pose_3d,
                'valid': valid_kp,
                'root_2d': root_2d,
                'root_3d': root_3d,
                'scale_factor': [w, h],
                'bbox': bbox
            }
            self.samples.append(item)
            if 'activity' in image_info:
                self.activities.add(image_info['activity'])
                self.image_activities.append(image_info['activity'])
                if image_info['activity'] not in self.activity_image_map:
                    self.activity_image_map[image_info['activity']] = []
                self.activity_image_map[image_info['activity']].append(idx)
        self.resampling()

    def resampling(self):
        sample_size = len(self.samples)
        all_indices = tuple(range(sample_size))
        if self.subset_percentage < 1:
            if self.sample_weight is not None:
                weight = self.sample_weight
            else:
                weight = [1] * sample_size
            weight /= np.sum(weight)
            self.indices = np.random.choice(
                all_indices,
                int(self.subset_percentage * sample_size),
                replace=False,
                p=weight
            )
        else:
            self.indices = all_indices

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx) -> dict:
        sample_idx = self.indices[idx]
        sample = self.samples[sample_idx]
        item = dict(
            img_id=sample['id'],
            keypoints_2d=sample['pose_2d'][:, :2].astype(np.float32),
            keypoints_3d=sample['pose_3d'].astype(np.float32),
        )
        if sample['valid'] is not None:
            item['valid'] = sample['valid']
        if sample['activity'] is not None:
            item['activity'] = sample['activity']
        return item
