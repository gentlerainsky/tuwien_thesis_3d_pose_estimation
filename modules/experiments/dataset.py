from torch.utils.data import DataLoader, sampler
from modules.lifter_2d_3d.dataset.base_dataset import BaseDataset
from modules.lifter_2d_3d.dataset.drive_and_act_keypoint_dataset import DriveAndActKeypointDataset
from modules.lifter_2d_3d.dataset.synthetic_cabin_ir_1m_dataset import SyntheticCabinIR1MKeypointDataset
from pathlib import Path

# common setting
# ------------
# dataset path
# ------------
drive_and_act_dataset_root_path = Path('/root/data/processed/drive_and_act/')
synthetic_cabin_ir_dataset_root_path = Path('/root/data/processed/synthetic_cabin_ir/')
synthetic_cabin_ir_1m_dataset_root_path = Path('/root/synthetic_cabin_1m/syntheticcabin_1mil/processed_syntheticCabin_1m/')
synthetic_cabin_ir_1m_v2_dataset_root_path = Path('/root/data/processed/synthetic_cabin_1m/')
synthetic_cabin_ir_1m_v2_image_root_path = Path('/root/synthetic_cabin_1m/syntheticcabin_1mil/SyntheticCabin_1m/')
# ------------
# model
# ------------
image_width = 1280
image_height = 1024
batch_size = 64

all_activities=[
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
    'opening_laptop',
    'reading_newspaper',
    'closing_laptop',
    'talking_on_phone',
    'closing_door_inside',
    'fastening_seat_belt',
    'placing_an_object',
    'fetching_an_object',
    'looking_or_moving_around (e.g. searching)',
    'preparing_food',
    'opening_door_inside',
    'opening_backpack',
    'entering_car',
    'taking_laptop_from_backpack',
    'putting_on_jacket',
    'taking_off_jacket',
    'exiting_car',
    'putting_laptop_into_backpack'
]


all_train_actors = ['vp1', 'vp2', 'vp3', 'vp4', 'vp5', 'vp6', 'vp7', 'vp8']
all_val_actors = ['vp9', 'vp10', 'vp15']
all_test_actors = ['vp11', 'vp12', 'vp13', 'vp14']


def construct_drive_and_act_dataset(
    dataset_name='drive_and_act',
    dataset_root_path=drive_and_act_dataset_root_path,
    viewpoint='a_column_co_driver',
    keypoint_2d_folder='keypoint_detection_results',
    keypoint_3d_folder='annotations',
    bbox_folder='person_detection_results',
    image_width=image_width,
    image_height=image_height,
    train_actors=all_train_actors,
    val_actors=all_val_actors,
    test_actors=all_test_actors,
    batch_size=batch_size,
    remove_activities=[],
    subset_percentage=100
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
        # is_normalize_to_bbox=False,
        # is_normalize_to_pose=True,
        # is_normalize_rotation=False,
        remove_activities=remove_activities,
        subset_percentage=subset_percentage
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
        # is_normalize_to_bbox=False,
        # is_normalize_to_pose=True,
        # is_normalize_rotation=False,
        subset_percentage=subset_percentage
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
        # is_center_to_neck=True,
        # is_normalize_to_bbox=False,
        # is_normalize_to_pose=True,
        is_normalize_rotation=False
    )
    train_dataset.make_weights_for_balanced_classes()
    # weighted_sampler = sampler.WeightedRandomSampler(train_dataset.sample_weight, len(train_dataset.sample_weight), replacement=True)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=24, sampler=weighted_sampler)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=24)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=24)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, num_workers=24, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=24
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=24
    )
    all_activities = train_dataset.activities.union(val_dataset.activities)\
        .union(test_dataset.activities)
    all_activities = train_dataset.activities.union(val_dataset.activities).union(test_dataset.activities)
    return dict(
        dataset_name=dataset_name,
        datasubset_name=viewpoint,
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
    subset_percentage=100
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
        is_normalize_rotation=False,
        is_gt_2d_pose=is_gt_2d_pose,
        subset_percentage=subset_percentage
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
        is_normalize_rotation=False,
        is_gt_2d_pose=is_gt_2d_pose,
        subset_percentage=subset_percentage
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
        is_normalize_rotation=False,
        is_gt_2d_pose=is_gt_2d_pose
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, num_workers=24, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=24
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


def construct_synthetic_cabin_ir_1m_v2(
    dataset_name='synthetic_cabin_ir_1m',
    dataset_root_path=synthetic_cabin_ir_1m_v2_dataset_root_path,
    viewpoints=None,
    keypoint_2d_folder='keypoint_detection_results',
    keypoint_2d_file_prefix='keypoint_detection',
    bbox_file_predix='human_detection',
    keypoint_3d_folder='annotations',
    bbox_folder='person_detection_results',
    image_width=image_width,
    image_height=image_height,
    batch_size=batch_size,
    is_gt_2d_pose=False,
    subset_percentage=100
):
    keypoint_2d_path = dataset_root_path / 'all_views' / keypoint_2d_folder
    keypoint_3d_path = dataset_root_path / 'all_views' / keypoint_3d_folder
    bbox_path = dataset_root_path / 'all_views' / bbox_folder
    train_dataset = SyntheticCabinIR1MKeypointDataset(
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
        is_normalize_rotation=False,
        is_gt_2d_pose=is_gt_2d_pose,
        included_view=viewpoints,
        subset_percentage=subset_percentage
    )
    val_dataset = SyntheticCabinIR1MKeypointDataset(
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
        is_normalize_rotation=False,
        is_gt_2d_pose=is_gt_2d_pose,
        included_view=viewpoints,
        subset_percentage=subset_percentage
    )
    test_dataset = SyntheticCabinIR1MKeypointDataset(
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
        is_normalize_rotation=False,
        is_gt_2d_pose=is_gt_2d_pose,
        included_view=viewpoints
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, drop_last=True, num_workers=24, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=24
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=24
    )
    all_activities = train_dataset.activities.union(val_dataset.activities)\
        .union(test_dataset.activities)
    subset_suffix = 'gt' if (keypoint_2d_folder == 'annotations') else 'predicted'
    return dict(
        dataset_name=dataset_name,
        datasubset_name=f'{subset_suffix}',
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        all_activities=all_activities
    )
