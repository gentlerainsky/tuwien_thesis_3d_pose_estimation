from torch.utils.data import DataLoader, sampler
from modules.lifter_2d_3d.dataset.drive_and_act_keypoint_dataset import DriveAndActKeypointDataset
from modules.lifter_2d_3d.dataset.base_dataset import BaseDataset
from pathlib import Path

# common setting
# ------------
# dataset path
# ------------
drive_and_act_dataset_root_path = Path('/root/data/processed/drive_and_act/')
synthetic_cabin_ir_dataset_root_path = Path('/root/data/processed/synthetic_cabin_ir/')
synthetic_cabin_ir_1m_dataset_root_path = Path('/root/synthetic_cabin_1m/syntheticcabin_1mil/processed_syntheticCabin_1m/')
# ------------
# model
# ------------
image_width = 1280
image_height = 1024
batch_size = 64


def construct_drive_and_act_dataset(
    dataset_root_path=drive_and_act_dataset_root_path,
    viewpoint='A_Pillar_Codriver',
    keypoint_2d_folder='keypoint_detection_results',
    keypoint_3d_folder='annotations',
    bbox_folder='person_detection_results',
    image_width=image_width,
    image_height=image_height,
    train_actors=['vp1', 'vp2', 'vp3', 'vp4', 'vp5', 'vp6', 'vp7', 'vp8'],
    val_actors=['vp9', 'vp10', 'vp15'],
    test_actors=['vp11', 'vp12', 'vp13', 'vp14'],
    batch_size=batch_size,
):
    keypoint_2d_path = dataset_root_path / viewpoint / keypoint_2d_folder
    keypoint_3d_path = dataset_root_path / viewpoint / keypoint_3d_folder
    bbox_path = dataset_root_path / viewpoint / bbox_folder
    train_dataset = DriveAndActKeypointDataset(
        prediction_file=(keypoint_2d_path / 'keypoint_detection_train.json').as_posix(),
        annotation_file=(keypoint_3d_path / 'person_keypoints_train.json').as_posix(),
        bbox_file=(bbox_path / 'human_detection_train.json').as_posix(),
        image_width=image_width,
        image_height=image_height,
        actors=train_actors,
        exclude_ankle=True,
        exclude_knee=True,
        bbox_format='xyxy',
        is_center_to_neck=True,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=True,
        is_normalize_rotation=True,
        # remove_activities=[
        #     'eating',
        #     'pressing_automation_button',
        #     'opening_bottle',
        #     'drinking',
        #     'interacting_with_phone',
        #     'closing_bottle',
        #     'reading_magazine',
        #     'sitting_still',
        #     'working_on_laptop',
        #     'putting_on_sunglasses',
        #     'taking_off_sunglasses',
        #     'unfastening_seat_belt',
        #     'using_multimedia_display',
        #     'writing',
        #     'opening_laptop',
        #     'reading_newspaper',
        #     'closing_laptop',
        #     'talking_on_phone',
        #     'closing_door_inside',
        #     'fastening_seat_belt',
        #     'placing_an_object',
        #     'fetching_an_object',
        #     'looking_or_moving_around (e.g. searching)',
        #     'preparing_food',
        #     'opening_door_inside',
        #     'opening_backpack',
        #     'entering_car',
        #     'taking_laptop_from_backpack',
        #     'putting_on_jacket',
        #     'taking_off_jacket',
        #     'exiting_car',
        #     'putting_laptop_into_backpack'
        # ]
    )
    val_dataset = DriveAndActKeypointDataset(
        prediction_file=(keypoint_2d_path / 'keypoint_detection_train.json').as_posix(),
        annotation_file=(keypoint_3d_path / 'person_keypoints_train.json').as_posix(),
        bbox_file=(bbox_path / 'human_detection_train.json').as_posix(),
        image_width=image_width,
        image_height=image_height,
        actors=val_actors,
        exclude_ankle=True,
        exclude_knee=True,
        bbox_format='xyxy',
        is_center_to_neck=True,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=True,
        is_normalize_rotation=True
    )
    test_dataset = DriveAndActKeypointDataset(
        prediction_file=(keypoint_2d_path / 'keypoint_detection_train.json').as_posix(),
        annotation_file=(keypoint_3d_path / 'person_keypoints_train.json').as_posix(),
        bbox_file=(bbox_path / 'human_detection_train.json').as_posix(),
        image_width=image_width,
        image_height=image_height,
        actors=test_actors,
        exclude_ankle=True,
        exclude_knee=True,
        bbox_format='xyxy',
        is_center_to_neck=True,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=True,
        is_normalize_rotation=True
    )
    train_dataset.make_weights_for_balanced_classes()
    weighted_sampler = sampler.WeightedRandomSampler(train_dataset.sample_weight, len(train_dataset.sample_weight), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=24, sampler=weighted_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=24)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=24)
    all_activities = train_dataset.activities.union(val_dataset.activities).union(test_dataset.activities)
    subset_suffix = 'gt' if (keypoint_2d_folder == 'annotations') else 'predicted'
    return dict(
        dataset_name='drive_and_act',
        datasubset_name=f'{viewpoint}_{subset_suffix}',
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        all_activities=all_activities
    )


def construct_synthetic_cabin_ir(
    dataset_name='synthetic_cabin_ir',
    dataset_root_path=synthetic_cabin_ir_dataset_root_path,
    viewpoint='A_Pillar_Codriver',
    keypoint_2d_folder='keypoint_detection_results',
    keypoint_2d_file_prefix='keypoint_detection',
    bbox_file_predix='human_detection',
    keypoint_3d_folder='annotations',
    bbox_folder='person_detection_results',
    image_width=image_width,
    image_height=image_height,
    batch_size=batch_size,
    is_gt_2d_pose=False,
):
    keypoint_2d_path = dataset_root_path / viewpoint / keypoint_2d_folder
    keypoint_3d_path = dataset_root_path / viewpoint / keypoint_3d_folder
    bbox_path = dataset_root_path / viewpoint / bbox_folder
    train_dataset = BaseDataset(
        prediction_file=(keypoint_2d_path / f'{keypoint_2d_file_prefix}_train.json').as_posix(),
        annotation_file=(keypoint_3d_path / 'person_keypoints_train.json').as_posix(),
        bbox_file=(bbox_path / f'{bbox_file_predix}_train.json').as_posix(),
        image_width=image_width,
        image_height=image_height,
        exclude_ankle=True,
        exclude_knee=True,
        bbox_format='xyxy',
        is_center_to_neck=True,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=True,
        is_normalize_rotation=True,
        is_gt_2d_pose=is_gt_2d_pose
    )
    val_dataset = BaseDataset(
        prediction_file=(keypoint_2d_path / f'{keypoint_2d_file_prefix}_val.json').as_posix(),
        annotation_file=(keypoint_3d_path / 'person_keypoints_val.json').as_posix(),
        bbox_file=(bbox_path / f'{bbox_file_predix}_val.json').as_posix(),
        image_width=image_width,
        image_height=image_height,
        exclude_ankle=True,
        exclude_knee=True,
        bbox_format='xyxy',
        is_center_to_neck=True,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=True,
        is_normalize_rotation=True,
        is_gt_2d_pose=is_gt_2d_pose
    )
    test_dataset = BaseDataset(
        prediction_file=(keypoint_2d_path / f'{keypoint_2d_file_prefix}_test.json').as_posix(),
        annotation_file=(keypoint_3d_path / 'person_keypoints_test.json').as_posix(),
        bbox_file=(bbox_path / f'{bbox_file_predix}_test.json').as_posix(),
        image_width=image_width,
        image_height=image_height,
        exclude_ankle=True,
        exclude_knee=True,
        bbox_format='xyxy',
        is_center_to_neck=True,
        is_normalize_to_bbox=False,
        is_normalize_to_pose=True,
        is_normalize_rotation=True,
        is_gt_2d_pose=is_gt_2d_pose
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, num_workers=24, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, drop_last=True, num_workers=24
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=24
    )
    all_activities = train_dataset.activities.union(val_dataset.activities)\
        .union(test_dataset.activities)
    subset_suffix = 'gt' if (keypoint_2d_folder == 'annotations') else 'predicted'
    return dict(
        dataset_name=dataset_name,
        datasubset_name=f'{viewpoint}_{subset_suffix}',
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        all_activities=all_activities
    )
