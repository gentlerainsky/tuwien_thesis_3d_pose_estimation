# score is the average of the confidences of all keypoints multiplied by person detected box's score
# hope this can help you

from modules.pose_estimator_2d.pose_estimator_2d import PoseEstimator2D
import pathlib
import json
import datetime
import numpy as np
import os


def infer_2d_pose_estimation(
    dataset_root_path=f"/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/",
    config_path="modules/pose_estimator_2d/config/hrnet.py",
    pretrained_path="https://download.openmmlab.com/mmpose/v1"
    "/body_2d_keypoint/topdown_heatmap/coco"
    "/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth",
    checkpoint_path="mmengine_workdir/pose_estimator_2d/best_coco_AP_epoch_9.pth",
    device="cuda:0",
    working_directory="mmengine_workdir/pose_estimator_2d",
    log_level="INFO",
    use_ground_truth_bbox=False,
    bbox_format='xyxy'
):
    dataset_root = pathlib.Path(dataset_root_path)
    if use_ground_truth_bbox:
        pose_estimator_2d = PoseEstimator2D(
            config_path=config_path,
            pretrained_path=pretrained_path,
            checkpoint_path=checkpoint_path,
            data_root_path=dataset_root_path,
            device=device,
            working_directory=working_directory,
            log_level=log_level,
            use_groundtruth_bbox=True,
        )
    else:
        pose_estimator_2d = PoseEstimator2D(
            config_path=config_path,
            pretrained_path=pretrained_path,
            checkpoint_path=checkpoint_path,
            data_root_path=dataset_root_path,
            device=device,
            working_directory=working_directory,
            log_level=log_level,
        )

    pose_estimator_2d.load_pretrained()
    image_sets = ["train", "val", "test"]
    for image_set in image_sets:
        results = []
        image_folder = dataset_root / "images" / image_set
        image_paths = list(image_folder.iterdir())
        count = 0
        max_count = len(image_paths)
        if use_ground_truth_bbox:
            bbox_detection_path = (
                dataset_root
                / "person_detection_results"
                / f"ground_truth_human_detection_{image_set}.json"
            )
        else:
            bbox_detection_path = (
                dataset_root
                / "person_detection_results"
                / f"human_detection_{image_set}.json"
            )
        with bbox_detection_path.open() as f:
            bboxes = json.loads(f.read())
        annotation_path = (
            dataset_root / "annotations" / f"person_keypoints_{image_set}.json"
        )
        with annotation_path.open() as f:
            annotations = json.loads(f.read())
        annotation_info = {}
        for annotation in annotations["images"]:
            annotation_info[annotation["id"]] = annotation
        start_time = datetime.datetime.now()
        print(f"\timage_set = {image_set}, start at = {start_time}")
        for bbox_info in bboxes:
            bbox = np.array(bbox_info["bbox"]).reshape(1, -1).astype(np.float32)
            image_filename = annotation_info[bbox_info["image_id"]]["file_name"]
            image_path = dataset_root / "images" / image_set / image_filename
            pose_estimator_2d_result = pose_estimator_2d.inference(
                image_path.as_posix(), bbox, bbox_format=bbox_format
            )
            keypoint_scores = pose_estimator_2d_result[0].pred_instances[
                "keypoint_scores"
            ]
            score = np.mean(keypoint_scores * bbox_info["score"])
            keypoints = pose_estimator_2d_result[0].pred_instances["keypoints"]
            keypoint_result = np.dstack([keypoints, np.expand_dims(keypoint_scores, 2)])
            result = {
                "image_id": int(bbox_info["image_id"]),
                "category_id": 1,
                "keypoints": keypoint_result.astype(float)
                    .round(decimals=4)
                    .reshape(-1)
                    .tolist(),
                "score": float(score),
                "scale": pose_estimator_2d_result[0]
                    .gt_instances["bbox_scales"][0]
                    .astype(float)
                    .round(decimals=4)
                    .tolist(),
                # "center": pose_estimator_2d_result[0]
                #     .gt_instances["bbox_centers"][0]
                #     .astype(float)
                #     .round(decimals=4)
                #     .tolist(),
            }
            results.append(result)
            count += 1
            if (count + 1) % 500 == 0:
                print(
                    f"\t\ttime = {datetime.datetime.now()}\n"
                    f"\t\t\tprogress = {count + 1}/{max_count} "
                    f"= {(count + 1)/max_count:.4f}"
                )

        end_time = datetime.datetime.now()
        total_secs = (end_time - start_time).total_seconds()
        total_mins = total_secs / 60
        print(f"\t\timage_set = {image_set}, finish at = {datetime.datetime.now()}")
        print(
            f"\t\ttime spend = {total_mins:.2f} mins ({total_secs:.2f} secs), fps = {(count + 1)/total_secs:.4f} fps"
        )

        result_folder = dataset_root / "keypoint_detection_results"
        if not result_folder.exists():
            print(f"created a folder at {str(result_folder)}.")
            os.makedirs(str(result_folder))
        if use_ground_truth_bbox:
            output_file = (
                result_folder
                / f"keypoint_detection_with_ground_truth_bbox_{image_set}.json"
            )
        else:
            output_file = result_folder / f"keypoint_detection_{image_set}.json"
        with output_file.open("w") as f:
            json.dump(results, f, indent=2)
