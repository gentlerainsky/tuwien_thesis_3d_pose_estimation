# score is the average of the confidences of all keypoints multiplied by person detected box's score
# hope this can help you

from src.modules.pose_estimator_2d.pose_estimator_2d import PoseEstimator2D
import pathlib
import json
import datetime
import numpy as np

# def finetune():
#     human_detector = HumanDetector(
#         config_path='./src/modules/human_detector/config/faster_rcnn.py',
#         pretrained_path='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'\
#             'faster_rcnn_r101_caffe_fpn_1x_coco/faster_rcnn_r101_caffe_fpn_1x_coco_bbox_mAP-0.398_20200504_180057-b269e9dd.pth',
#         checkpoint_path='./mmengine_workdir/human_detector/epoch_1.pth',
#         data_root_path='/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/',
#         device='cuda:0',
#         working_directory='./mmengine_workdir/human_detector',
#         log_level='CRITICAL'
#     )
#     human_detector.finetune()


def infer_2d_pose_estimation():
    pose_estimator_2d = PoseEstimator2D(
        config_path='src/modules/pose_estimator_2d/config/hrnet.py',
        pretrained_path='https://download.openmmlab.com/mmpose/v1' \
            '/body_2d_keypoint/topdown_heatmap/coco'\
                '/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth',
        # checkpoint_path="mmengine_workdir/pose_estimator_2d/best_coco_AP_epoch_0.pth",
        checkpoint_path="mmengine_workdir/pose_estimator_2d/best_coco_AP_epoch_9.pth",
        data_root_path='/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/',
        device='cuda:0',
        working_directory='mmengine_workdir/pose_estimator_2d',
        log_level='INFO'
    )

    pose_estimator_2d.load_pretrained()
    dataset_root = pathlib.Path(f'/root/data/processed/synthetic_cabin_bw')
    camera_positions = list(dataset_root.iterdir())
    image_sets = ['train', 'val', 'test']
    for camera_position_folder in camera_positions[2:]:
        # for image_set in image_sets[0:1]:
        print(f'camera_position = {camera_position_folder.name}')

        for image_set in image_sets:
            results = []
            image_folder = camera_position_folder / 'images' / image_set
            image_paths = list(image_folder.iterdir())
            count = 0
            max_count = len(image_paths)
            bbox_detection_path = dataset_root / camera_position_folder / 'person_detection_results' / f'human_detection_{image_set}.json'
            with bbox_detection_path.open() as f:
                bboxes = json.loads(f.read())
            annotation_path = dataset_root / camera_position_folder / 'annotations' / f'person_keypoints_{image_set}.json'
            with annotation_path.open() as f:
                annotations = json.loads(f.read())
            annotation_info = {}
            for annotation in annotations['images']:
                annotation_info[annotation['id']] = annotation
            start_time = datetime.datetime.now()
            print(f'\timage_set = {image_set}, start at = {start_time}')
            for bbox_info in bboxes:
                bbox = np.array(bbox_info['bbox']).reshape(1, -1)
                image_filename = annotation_info[bbox_info['image_id']]['file_name']
                image_path = dataset_root / camera_position_folder / 'images' / image_set / image_filename
                pose_estimator_2d_result = pose_estimator_2d.inference(image_path, bbox, bbox_format='xywh')
                keypoint_scores = pose_estimator_2d_result[0].pred_instances['keypoint_scores']
                score = np.mean(keypoint_scores * bbox_info['score'])
                keypoints = pose_estimator_2d_result[0].pred_instances['keypoints']
                # print(keypoints.shape)
                # print(np.expand_dims(keypoint_scores, 2).shape)
                keypoint_result = np.dstack([keypoints, np.expand_dims(keypoint_scores, 2)])
                result = {
                    'image_id': int(bbox_info['image_id']),
                    'category_id': 1,
                    'keypoints': keypoint_result.astype(float).round(decimals=4).reshape(-1).tolist(),
                    'score': float(score),
                    'scale': pose_estimator_2d_result[0].gt_instances['bbox_scales'][0].astype(float).round(decimals=4).tolist(),
                    'center': pose_estimator_2d_result[0].gt_instances['bbox_centers'][0].astype(float).round(decimals=4).tolist()
                }
                # print(f'result\n{result}')
                results.append(result)
                count += 1
                if (count + 1) % 500 == 0:
                    print(
                        f'\t\ttime = {datetime.datetime.now()}\n'
                        f'\t\t\tprogress = {count + 1}/{max_count} '
                            f'= {(count + 1)/max_count:.4f}'
                    )

                # if (count + 1) % 10 == 0:
                    # break
            end_time = datetime.datetime.now()
            total_secs = (end_time - start_time).total_seconds()
            total_mins = total_secs / 60
            print(f'\t\timage_set = {image_set}, finish at = {datetime.datetime.now()}')
            print(f'\t\ttime spend = {total_mins:.2f} mins ({total_secs:.2f} secs), fps = {(count + 1)/total_secs:.4f} fps')
            with (camera_position_folder / 'keypoint_detection_results' / f'keypoint_detection_{image_set}.json').open('w') as f:
                json.dump(results, f, indent=2)


if __name__ == '__main__':
    pass
