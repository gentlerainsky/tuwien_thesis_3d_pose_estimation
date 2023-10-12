import json
import glob
import os
from pathlib import Path
import cv2
import copy
import pandas as pd


annotation_template = {
    "info": {
        "description": "Drive&Act Dataset",
        "url": "",
        "version": "1.0",
        "year": 2021,
        "contributor": "Manuel Martin, Alina Roitberg, Monica Haurilet, Matthias Horne, Simon Reiß, Michael Voit, Rainer Stiefelhagen",
        "date_created": "2019/04/17",
    },
    "licenses": [
        {"url": "https://driveandact.com/", "id": 1, "name": "Fraunhofer IOSB"}
    ],
    "images": [],
    "annotations": [],
    "camera_parameters": {},
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose",
                "left_eye",
                "right_eye",
                "left_ear",
                "right_ear",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ],
            "skeleton": [
                [16, 14],
                [14, 12],
                [17, 15],
                [15, 13],
                [12, 13],
                [6, 12],
                [7, 13],
                [6, 7],
                [6, 8],
                [7, 9],
                [8, 10],
                [9, 11],
                [2, 3],
                [1, 2],
                [1, 3],
                [2, 4],
                [3, 5],
                [4, 6],
                [5, 7],
            ],
        }
    ],
}

all_drive_and_act_keypoint_names = [
    'nose', 'lElbow', 'lWrist', 'rHeel', 'rHip',
    'rSmallToe', 'neck', 'lSmallToe', 'rWrist',
    'rAnkle', 'lHip', 'lHeel', 'lKnee', 'lEye',
    'midHip', 'background', 'lEar', 'rElbow',
    'rShoulder', 'rKnee', 'lShoulder', 'lBigToe',
    'rEye', 'rEar', 'rBigToe', 'lAnkle'
]
drive_and_act_keypoint_names = [
    "nose",
    "lEye",
    "rEye",
    "lEar",
    "rEar",
    "lShoulder",
    "rShoulder",
    "lElbow",
    "rElbow",
    "lWrist",
    "rWrist",
    "lHip",
    "rHip",
    "lKnee",
    "rKnee",
    "lAnkle",
    "rAnkle",
]
coco_keypoint_names = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


class DriveAndActPreprocessor:
    def __init__(
            self,
            dataset_root,
            video_root,
            source_annotation_3d_root,
            destination_root,
            image_folder,
            train_image_folder,
            val_image_folder,
            test_image_folder,
            destination_annotation_folder,
            val_subset,
            test_subset,
            start_frame_id=0,
            sampling_rate=5
        ):
        self.dataset_root = Path(dataset_root)
        self.video_root = self.dataset_root / video_root
        self.source_annotation_3d_root = self.dataset_root / source_annotation_3d_root
        self.destination_root = Path(destination_root)
        self.image_folder = self.destination_root / image_folder
        self.train_image_path = self.image_folder / train_image_folder
        self.val_image_path = self.image_folder / val_image_folder
        self.test_image_path = self.image_folder / test_image_folder
        self.train_image_annotations = []
        self.val_image_annotations = []
        self.test_image_annotations = []
        self.train_keypoint_annotations = []
        self.val_keypoint_annotations = []
        self.test_keypoint_annotations = []
        self.sampling_rate = sampling_rate
        self.destination_annotation_folder = self.destination_root / destination_annotation_folder
        self.val_subset = val_subset
        self.test_subset = test_subset
        self.destination_height = 1024
        self.destination_width = 1280
        self.frame_id = start_frame_id
        self.create_folder_if_not_exist()

    def create_folder_if_not_exist(self):
        paths = [
            self.destination_root,
            self.image_folder,
            self.train_image_path,
            self.val_image_path,
            self.test_image_path,
            self.destination_annotation_folder,
        ]
        for path in paths:
            if not path.exists():
                print(f'created a folder at {str(path)}.')
                os.makedirs(str(path))

    def write_annotation_file(self):
        train_info = copy.deepcopy(annotation_template)
        val_info = copy.deepcopy(annotation_template)
        test_info = copy.deepcopy(annotation_template)
        with (self.destination_annotation_folder / 'person_keypoints_train.json').open('w') as outfile:
            train_info['images'] = self.train_image_annotations
            train_info['annotations'] = self.train_keypoint_annotations
            json.dump(train_info, outfile)

        with (self.destination_annotation_folder / 'person_keypoints_val.json').open('w') as outfile:
            val_info['images'] = self.val_image_annotations
            val_info['annotations'] = self.val_keypoint_annotations
            json.dump(val_info, outfile)
        
        with (self.destination_annotation_folder / 'person_keypoints_test.json').open('w') as outfile:
            test_info['images'] = self.test_image_annotations
            test_info['annotations'] = self.test_keypoint_annotations
            json.dump(test_info, outfile)

    def extract_all(self):
        actors = [item for item in self.video_root.iterdir() if item.is_dir()]
        for actor in actors:
            self.extract_frame_by_actor(actor.name)
            break
        self.write_annotation_file()

    def extract_frame_by_actor(self, actor):
        actor_folder = self.video_root / actor
        video_files = [file for file in actor_folder.iterdir() if file.name.split('.')[-1] == 'mp4']
        # camera_parameter_files = [file for file in actor_folder.iterdir() if file.name.split('.')[-1] == 'json']
        annotation_folders = self.source_annotation_3d_root / actor
        annotation_files = [file for file in annotation_folders.iterdir() if file.name.split('.')[-1] == 'csv']
        for video_file in video_files:
            file_prefix = video_file.name.split('.')[:-1]
            image_prefix = '_'.join(file_prefix)
            match_annotation_files = [
                file for file in annotation_files if file.name.split('.')[0] == file_prefix[0]
            ]
            # match_camera_parameter_files = [
                # file for file in camera_parameter_files if file.name.split('.')[0] == file_prefix[0]
            # ]
            if len(match_annotation_files) > 0:                
                self.extract_frame_by_video(
                    actor,
                    image_prefix,
                    video_file,
                    match_annotation_files[0],
                    # match_camera_parameter_files[0]
                )
            break

    def create_final_image_annotations(self, frame_id, filename):
        image_annotation = {
            "license": 1,
            "file_name": filename,
            "coco_url": "",
            "height": self.destination_height,
            "width": self.destination_width,
            "data_captured": "",
            "flickr_url": "",
            "id": frame_id,
        }
        return image_annotation


    def create_final_skeleton_annotaions(self, frame_id, num_keypoints, keypoints3D):
        skeleton_annotation = {
            # "segmentation": [],
            "num_keypoints": num_keypoints,
            # "area": round(humans["full_body_bbox"]["bbox_area"] * self.scale_factor, 4),
            # "area": 0,
            "iscrowd": 0,
            # "keypoints": self.calc_keypoints(humans["keypoints2D"]),
            # "keypoints": [],
            "keypoints3D": keypoints3D,
            "image_id": frame_id,
            # "bbox": self.calc_bbox(humans["full_body_bbox"]),
            # "bbox": [],
            "category_id": 1,
            "id": frame_id,
        }
        return skeleton_annotation

    def get_image_folder_path(self, actor):
        if actor in self.val_subset:
            return self.val_image_path
        elif actor in self.test_subset:
            return self.test_image_path
        return self.train_image_path

    def extract_frame_by_video(self, actor, image_prefix, video_file, annotation_file):
        annotation_df = pd.read_csv(annotation_file.as_posix())
        # print(annotation_df.head())
        # check if the annotation exists at a particular keypoint
        # to remove unnecessary point like foot toes.
        annotation_df.insert(2, "is_valid", (annotation_df.iloc[:, 2:].sum(axis=1) > 0))

        image_annotations = []
        keypoint_annotations = []
        # get annotation column name (keypoint name)
        # l = annotation_df.columns[3:].map(lambda a: a.split('_')[0])
        # keypoint_name = []
        # prev_name = None
        # for name in l:
        #     if name != prev_name:
        #         keypoint_name.append(name)
        #         prev_name = name
        cap = cv2.VideoCapture(video_file.as_posix())
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        destination_image_folder_path = self.get_image_folder_path(actor)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        print('frame_rate', frame_rate)
        for frame_number in range(0, total_frames, frame_rate):
            # avoid frames with no actor
            if not annotation_df.iloc[frame_number].is_valid:
                continue
            
            pose_3d = pd.DataFrame(
                annotation_df.iloc[frame_number][annotation_df.columns[3:]].to_numpy().reshape([-1, 4]),
                index=all_drive_and_act_keypoint_names, columns=['x', 'y', 'z', 'p']
            )
            num_keypoints = int((pose_3d.loc[drive_and_act_keypoint_names]['p'] > 0).sum())
            pose_3d_array = pose_3d.loc[drive_and_act_keypoint_names][['x', 'y', 'z']].values.reshape(-1).tolist()

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            _, image = cap.read()
            image_file_name = f'{image_prefix}_frame_{frame_number}.jpg'
            image_path = destination_image_folder_path / image_file_name
            image_annotation = self.create_final_image_annotations(self.frame_id, image_file_name)
            image_annotations.append(image_annotation)
            skeleton_annotation = self.create_final_skeleton_annotaions(self.frame_id, num_keypoints, pose_3d_array)
            keypoint_annotations.append(skeleton_annotation)
            self.frame_id += 1
            cv2.imwrite(image_path.as_posix(), image)
        
        if actor in self.val_subset:
            self.val_image_annotations += image_annotations
            self.val_keypoint_annotations += keypoint_annotations
        elif actor in self.test_subset:
            self.test_image_annotations += image_annotations
            self.test_keypoint_annotations += keypoint_annotations
        else:
            self.train_image_annotations += image_annotations
            self.train_keypoint_annotations += keypoint_annotations
