from torch.utils.data import DataLoader, sampler
from modules.lifter_2d_3d.dataset.drive_and_act_keypoint_dataset import DriveAndActKeypointDataset
from pathlib import Path

# common setting
# ------------
# dataset path
# ------------
dataset_root_path = Path('/root/data/processed/drive_and_act/')
keypoint_2d_path = dataset_root_path / 'keypoint_detection_results'
keypoint_3d_path = dataset_root_path / 'annotations'
bbox_file = dataset_root_path / 'person_detection_results'
# ------------
# model
# ------------
image_width = 1280
image_height = 1024
batch_size = 64


def construct_drive_and_act_dataset(
        keypoint_2d_path=keypoint_2d_path,
        keypoint_3d_path=keypoint_3d_path,
        bbox_file=bbox_file,
        image_width=image_width,
        image_height=image_height,
        train_actors=['vp1', 'vp2', 'vp3', 'vp4', 'vp5', 'vp6', 'vp7', 'vp8'],
        val_actors=['vp9', 'vp10', 'vp15'],
        test_actors=['vp11', 'vp12', 'vp13', 'vp14'],
        batch_size=batch_size,
):
    train_dataset = DriveAndActKeypointDataset(
        prediction_file=(keypoint_2d_path / 'keypoint_detection_train.json').as_posix(),
        annotation_file=(keypoint_3d_path / 'person_keypoints_train.json').as_posix(),
        bbox_file=(bbox_file / 'human_detection_train.json').as_posix(),
        image_width=image_width,
        image_height=image_height,
        actors=train_actors,
        exclude_ankle=True,
        exclude_knee=True,
        is_center_to_neck=True,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=True,
        is_normalize_rotation=True,
        remove_activities=[
            'eating',
            'pressing_automation_button',
            'opening_bottle',
            'drinking',
            'interacting_with_phone',
            'closing_bottle',
            'reading_magazine',
            'sitting_still',
            'working_on_laptop',
            'putting_on_sunglasses',
            'taking_off_sunglasses',
            'unfastening_seat_belt',
            'using_multimedia_display',
            'writing',
            # 'opening_laptop',
            # 'reading_newspaper',
            # 'closing_laptop',
            # 'talking_on_phone',
            # 'closing_door_inside',
            # 'fastening_seat_belt',
            # 'placing_an_object',
            # 'fetching_an_object',
            # 'looking_or_moving_around (e.g. searching)',
            # 'preparing_food',
            # 'opening_door_inside',
            # 'opening_backpack',
            # 'entering_car',
            # 'taking_laptop_from_backpack',
            # 'putting_on_jacket',
            # 'taking_off_jacket',
            # 'exiting_car',
            # 'putting_laptop_into_backpack'
        ]
    )
    val_dataset = DriveAndActKeypointDataset(
        prediction_file=(keypoint_2d_path / 'keypoint_detection_train.json').as_posix(),
        annotation_file=(keypoint_3d_path / 'person_keypoints_train.json').as_posix(),
        bbox_file=(bbox_file / 'human_detection_train.json').as_posix(),
        image_width=image_width,
        image_height=image_height,
        actors=val_actors,
        exclude_ankle=True,
        exclude_knee=True,
        is_center_to_neck=True,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=True,
        is_normalize_rotation=True
    )
    test_dataset = DriveAndActKeypointDataset(
        prediction_file=(keypoint_2d_path / 'keypoint_detection_train.json').as_posix(),
        annotation_file=(keypoint_3d_path / 'person_keypoints_train.json').as_posix(),
        bbox_file=(bbox_file / 'human_detection_train.json').as_posix(),
        image_width=image_width,
        image_height=image_height,
        actors=test_actors,
        exclude_ankle=True,
        exclude_knee=True,
        is_center_to_neck=True,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=True,
        is_normalize_rotation=True
    )
    weighted_sampler = sampler.WeightedRandomSampler(train_dataset.sample_weight, len(train_dataset.sample_weight), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=24, sampler=weighted_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=24)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=24)
    all_activities = train_dataset.activities.union(val_dataset.activities).union(test_dataset.activities)
    return dict(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        all_activities=all_activities
    )

