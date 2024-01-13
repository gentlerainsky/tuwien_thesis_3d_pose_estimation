# score is the average of the confidences of all keypoints multiplied by person detected box's score
# hope this can help you

from modules.pose_estimator_2d.pose_estimator_2d import PoseEstimator2D
from mmpose.apis import MMPoseInferencer
import pathlib
import json
import datetime
import numpy as np
import os
import cv2


def infer_2d_pose_estimation(
    dataset_root_path,
    mmpose_config_path,
    mmpose_model_weight,
    mmdet_config_path,
    mmdet_model_weight,
    device,
    det_cat_ids=[0]
):
    mmpose_inferencer = MMPoseInferencer(
        pose2d=mmpose_config_path,
        pose2d_weights=mmpose_model_weight,
        det_model=mmdet_config_path,
        det_weights=mmdet_model_weight,
        det_cat_ids=det_cat_ids,
        device=device,
        show_progress=False
    )
    dataset_root = pathlib.Path(dataset_root_path)
    image_sets = ['train', 'val', 'test']
    for image_set in image_sets:
        det_results = []
        pose2d_results = []
        image_folder = dataset_root / 'images' / image_set
        image_paths = list(image_folder.iterdir())
        count = 0
        max_count = len(image_paths)
        start_time = datetime.datetime.now()
        print(f'\timage_set = {image_set}, start at = {start_time}')
        for image_path in image_paths:
            img_id = int(image_path.name.split('.')[0])
            img_path_str = str(image_path)
            img = cv2.imread(img_path_str)
            mmpose_result = next(mmpose_inferencer(img))
            image_results = mmpose_result['predictions'][0]
            # Get the highest score bbox
            image_results = list(image_results)
            # pose_confidence = [np.average(item['keypoint_scores']) for item in image_results]
            # best_box_idx = np.argmax(pose_confidence)
            best_box_idx = np.argmax([box['bbox_score'] for box in image_results])
            count += 1
            if (count + 1) % 500 == 0:
                print(
                    f'\t\ttime = {datetime.datetime.now()}\n'
                    f'\t\t\tprogress = {count + 1}/{max_count} '
                        f'= {(count + 1)/max_count:.4f}'
                )

            detected_result = image_results[best_box_idx]
            # BBOX
            bbox = detected_result['bbox']
            bbox_score = detected_result['bbox_score']
            x, y, x2, y2 = np.array(bbox[0]).round(decimals=4).tolist()
            w = x2 - x
            h = y2 - y

            det_result = {
                'image_id': img_id,
                'category_id': 1,
                'bbox': [float(x), float(y), float(w), float(h)],
                'score': round(float(bbox_score), 4)
            }
            det_results.append(det_result)
            # 2D Poses
            pose_2d = detected_result['keypoints']
            pose_score = detected_result['keypoint_scores']
            result_pose = np.append(np.array(pose_2d), np.array(pose_score)\
                                    .reshape(-1, 1), axis=1)
            result_pose = result_pose.reshape(-1).round(decimals=4).tolist()
            result_pose_score = np.mean(pose_score).round(decimals=4)
            pose2d_result = {
                'image_id': img_id,
                'category_id': 1,
                'keypoints': result_pose,
                'score': result_pose_score,
            }
            pose2d_results.append(pose2d_result)
        end_time = datetime.datetime.now()
        total_secs = (end_time - start_time).total_seconds()
        total_mins = total_secs / 60
        print(f'\t\timage_set = {image_set}, finish at = {datetime.datetime.now()}')
        print(f'\t\ttime spend = {total_mins:.2f} mins ({total_secs:.2f} secs), fps = {(count + 1)/total_secs:.4f} fps')
        
        det_result_folder = dataset_root / 'person_detection_results'
        if not det_result_folder.exists():
            print(f'created a folder at {str(det_result_folder)}.')
            os.makedirs(str(det_result_folder))
        with (det_result_folder / f'human_detection_{image_set}.json').open('w') as f:
            print(det_results)
            json.dump(det_results, f, indent=2)
        
        result_folder = dataset_root / 'keypoint_detection_results'
        if not result_folder.exists():
            print(f'created a folder at {str(result_folder)}.')
            os.makedirs(str(result_folder))
        output_file = result_folder / f'keypoint_detection_{image_set}.json'
        with output_file.open('w') as f:
            json.dump(pose2d_results, f, indent=2)
