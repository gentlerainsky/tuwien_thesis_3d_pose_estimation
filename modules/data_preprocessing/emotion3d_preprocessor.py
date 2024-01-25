import json
import glob
import os
from pathlib import Path
import cv2
import copy
from modules.data_preprocessing.definition import (
    coco_keypoint_names,
    coco_keypoint_connections
)


annotation_template = {
    "info": {
        "description": "emotion3D dataset for triangulation",
        "url": "",
        "version": "1.0",
        "year": 2021,
        "contributor": "emotion3D",
        "date_created": "2021/12/15",
    },
    "licenses": [{"url": "", "id": 1, "name": "emotion3D License"}],
    "images": [],
    "annotations": [],
    "camera_parameters": {},
    "categories": [
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": coco_keypoint_names,
            "skeleton": coco_keypoint_connections,
        }
    ],
}



class Emotion3DPreprocessor:
    def __init__(
        self,
        source_path,
        destination_path,
        source_annotation_folder,
        source_image_folder,
        annotation_folder,
        image_folder,
        train_image_folder,
        val_image_folder,
        test_image_folder,
        person_detection_folder,
        keypoint_detection_folder,
        visualization_folder,
        source_height,
        source_width,
        destination_height,
        destination_width,
        val_subset,
        test_subset,
        camera_positions=None,
        is_write_image=True
    ):
        self.source_path = Path(source_path)
        self.root_destination_path = Path(destination_path)
        self.source_image_path = self.source_path / source_image_folder
        self.source_annotation_path = self.source_path / source_annotation_folder
        self.annotation_folder = annotation_folder
        self.image_folder = image_folder
        self.train_image_folder = train_image_folder
        self.val_image_folder = val_image_folder
        self.test_image_folder = test_image_folder
        self.person_detection_folder = person_detection_folder
        self.keypoint_detection_folder = keypoint_detection_folder
        self.visualization_folder = visualization_folder
        self.source_height = source_height
        self.source_width = source_width
        self.destination_height = destination_height
        self.destination_width = destination_width
        self.scale_height = self.destination_height / self.source_height
        self.scale_width = self.destination_width / self.source_width
        self.scale_factor = self.scale_height * self.scale_width
        self.val_subset = val_subset
        self.test_subset = test_subset
        self.camera_positions = camera_positions
        self.is_write_image = is_write_image

    def set_camera_position(self, camera_position):
        self.destination_path = self.root_destination_path / camera_position
        self.annotation_path = self.destination_path / self.annotation_folder
        self.image_path = self.destination_path / self.image_folder
        self.train_image_path = self.image_path / self.train_image_folder
        self.val_image_path = self.image_path / self.val_image_folder
        self.test_image_path = self.image_path / self.test_image_folder
        self.person_detection_path = self.destination_path / self.person_detection_folder
        self.keypoint_detection_path = self.destination_path / self.keypoint_detection_folder
        self.visualization_path = self.destination_path / self.visualization_folder

    def create_folder_if_not_exist(self):
        paths = [
            self.annotation_path,
            self.train_image_path,
            self.val_image_path,
            self.test_image_path,
            self.person_detection_path,
            self.keypoint_detection_path,
            self.visualization_path
        ]
        for path in paths:
            if not path.exists():
                os.makedirs(str(path))

    def person_detection(self, final_annotations):
        detections = []
        for annotation in final_annotations:
            temp_person_detection = {
                "image_id": annotation["image_id"],
                "category_id": 1,  # What is this?
                "bbox": annotation["bbox"],
                "score": 1,
            }
            detections.append(temp_person_detection)
        return detections


    def save_images(self, img_name, cat):
        img_path = self.source_image_path / img_name
        img = cv2.imread(str(img_path))
        img_name = img_name.split("frame_")[1]
        img_name = "000000" + img_name
        if cat == "train":
            cv2.imwrite(str(self.train_image_path / img_name), img)
        elif cat == "test":
            cv2.imwrite(str(self.test_image_path / img_name), img)
        else:
            cv2.imwrite(str(self.val_image_path / img_name), img)
        # print(destination_path, camera_pos, img_name)
        # print(os.path.join(destination_path, camera_pos, "images/train", img_name))


    # filename = frame_000001.jpg , nr = 1
    def create_final_image_annotations(self, filename, file_id):
        image_annotation = {
            "license": 1,
            "file_name": filename,
            "coco_url": "",
            "height": self.destination_height,
            "width": self.destination_width,
            "data_captured": "2021-06-22",
            "flickr_url": "",
            "id": file_id,
        }
        return image_annotation


    def create_final_skeleton_annotaions(self, file_id, current_json, annotation_list):
        filename = current_json["img_name"]
        img_id = int((filename.split(".jpg")[0]).split("frame_")[1])

        skeleton_annotation = annotation_list
        for humans in current_json["humans"]:
            temp_skeleton_annotation = {
                "segmentation": [],
                "num_keypoints": self.calc_num_keypoints(humans["keypoints2D"]),
                "area": round(humans["full_body_bbox"]["bbox_area"] * self.scale_factor, 4),
                "iscrowd": 0,
                "keypoints": self.calc_keypoints(humans["keypoints2D"]),
                "keypoints3D": self.calc_keypoints3D(humans["keypoints3D"]),
                "image_id": img_id,
                "bbox": self.calc_bbox(humans["full_body_bbox"]),
                "category_id": 1,
                "id": file_id,
            }
            skeleton_annotation.append(temp_skeleton_annotation)

        return skeleton_annotation


    def calc_num_keypoints(self, keypoints):
        # Only the 17 COCO annotations
        keypoints = keypoints[0:17]
        num_keypoints = 0
        for keypoint in keypoints:
            if keypoint["visibility"] != 0:
                num_keypoints += 1

        return num_keypoints


    def calc_keypoints(self, keypoints):
        # Only the 17 COCO annotations
        keypoints = keypoints[0:17]
        coco_keypoints = [0] * 17
        coco_keypoints[1:5] = keypoints[0:4]
        coco_keypoints[0] = keypoints[4]
        coco_keypoints[5:17] = keypoints[5:17]

        formated_keypoints = []
        for keypoint in coco_keypoints:
            formated_keypoints.append(round(keypoint["x"] * self.scale_height, 2))
            formated_keypoints.append(round(keypoint["y"] * self.scale_width, 2))
            formated_keypoints.append(keypoint["visibility"])

        return formated_keypoints


    def calc_keypoints3D(self, keypoints):
        # Only the 17 COCO annotations
        keypoints = keypoints[0:17]
        coco_keypoints = [0] * 17
        coco_keypoints[1:5] = keypoints[0:4]
        coco_keypoints[0] = keypoints[4]
        coco_keypoints[5:17] = keypoints[5:17]

        formated_keypoints = []
        for keypoint in coco_keypoints:
            formated_keypoints.append(round(keypoint["x"] * self.scale_height, 2))
            formated_keypoints.append(round(keypoint["y"] * self.scale_width, 2))
            #### ! Problem with scaling z-axis
            formated_keypoints.append(round(keypoint["z"] * self.scale_width, 2))

        return formated_keypoints


    def calc_bbox(self, temp_bbox, gt_bbox=True):
        bbox = []
        if gt_bbox:
            x = round(temp_bbox["x"] * self.scale_height, 2)
            y = round(temp_bbox["y"] * self.scale_width, 2)
            w = round(temp_bbox["width"] * self.scale_width, 2)
            h = round(temp_bbox["height"] * self.scale_height, 2)
            # use xyxy format for mmdet/mmpose compatibility
            bbox = [x, y, x + w, x + h]
        else:
            bbox.append(0)
            bbox.append(0)
            bbox.append(self.destination_width)
            bbox.append(self.destination_height)

        return bbox

    def format_dataset(
        self, annotation_template, num_train=100, num_val=0, num_test=0, print_set=False
    ):
        # convert keypoints to COCO keypoints
        total_image_counter = 0
        train_counter = 0
        test_counter = 0
        val_counter = 0
        train_info = {}
        test_info = {}
        val_info = {}
        # train_json = copy.deepcopy(annotation_template)
        # test_json = copy.deepcopy(annotation_template)
        # val_json = copy.deepcopy(annotation_template)

        files = sorted(glob.iglob(str(self.source_annotation_path) + "/*.json"))
        print(str(self.source_annotation_path), len(files))
        for i, file_path in enumerate(files):
            f = open(file_path)
            data = json.load(f)
            img_name = data["img_name"]
            filename = data["img_name"].split("frame_")[1]
            filename = "000000" + filename
            file_id = int(filename.split(".jpg")[0])
            human_name = data["humans"][0]["name"]

            camera_position = data["camera_parameters"]["name"]
            enough_train = False or (num_train == 0)
            enough_val = False or (num_val == 0)
            enough_test = False or (num_test == 0)
            total_image_counter += 1
            if total_image_counter % 10000 == 0:
                print(total_image_counter, enough_train, num_train)
            if (self.camera_positions is not None) and (camera_position not in self.camera_positions):
                continue
            if camera_position not in train_info:
                print(f'init {camera_position}')
                train_info[camera_position] = copy.deepcopy(annotation_template)
                val_info[camera_position] = copy.deepcopy(annotation_template)
                test_info[camera_position] = copy.deepcopy(annotation_template)
                self.set_camera_position(camera_position)
                self.create_folder_if_not_exist()

            if human_name in self.val_subset:
                if (val_counter <= num_val) and num_val > 0:
                    val_info[camera_position]["images"].append(
                        self.create_final_image_annotations(filename, file_id)
                    )
                    val_info[camera_position]["annotations"] = self.create_final_skeleton_annotaions(
                        file_id, data, val_info[camera_position]["annotations"]
                    )
                    if i == 0:
                        val_info[camera_position]["camera_parameters"] = data["camera_parameters"]
                    val_counter += 1
                    if self.is_write_image:
                        self.save_images(img_name, "val")
                else:
                    enough_val = True
            elif human_name in self.test_subset:
                if (test_counter <= num_test) and num_test > 0:
                    test_info[camera_position]["images"].append(
                        self.create_final_image_annotations(filename, file_id)
                    )
                    test_info[camera_position]["annotations"] = self.create_final_skeleton_annotaions(
                        file_id, data, test_info[camera_position]["annotations"]
                    )
                    if i == 0:
                        test_info[camera_position]["camera_parameters"] = data["camera_parameters"]
                    test_counter += 1
                    if self.is_write_image: 
                        self.save_images(img_name, "test")
                else:
                    enough_test = True
            else:
                if (train_counter <= num_train) and num_train > 0:
                    train_info[camera_position]["images"].append(
                        self.create_final_image_annotations(filename, file_id)
                    )
                    train_info[camera_position]["annotations"] = self.create_final_skeleton_annotaions(
                        file_id, data, train_info[camera_position]["annotations"]
                    )
                    train_counter += 1
                    if i == 0:
                        train_info[camera_position]["camera_parameters"] = data["camera_parameters"]
                    if self.is_write_image:
                        self.save_images(img_name, "train")
                else:
                    enough_train = True
            if enough_train and enough_val and enough_test:
                break
        print('finish saving images')
        for camera_position in train_info.keys():
            self.set_camera_position(camera_position)
            
            # create json files for keypoints
            with (self.annotation_path / 'person_keypoints_train.json').open('w') as outfile:
                json.dump(train_info[camera_position], outfile, indent=2)

            with (self.annotation_path / 'person_keypoints_val.json').open('w') as outfile:
                json.dump(val_info[camera_position], outfile, indent=2)
            
            with (self.annotation_path / 'person_keypoints_test.json').open('w') as outfile:
                json.dump(test_info[camera_position], outfile, indent=2)
                
            with (self.person_detection_path / 'ground_truth_human_detection_train.json').open('w') as outfile:
                # person detection results
                detections_json = self.person_detection(train_info[camera_position]["annotations"])
                json.dump(detections_json, outfile, indent=2)
            
            with (self.person_detection_path / 'ground_truth_human_detection_val.json').open('w') as outfile:
                # person detection results
                detections_json = self.person_detection(val_info[camera_position]["annotations"])
                json.dump(detections_json, outfile, indent=2)

            with (self.person_detection_path / 'ground_truth_human_detection_test.json').open('w') as outfile:
                # person detection results
                detections_json = self.person_detection(test_info[camera_position]["annotations"])
                json.dump(detections_json, outfile, indent=2)


            # visualize keypoints
            if print_set:
                self.print_keypoints(train_info, 'train', camera_position)

if __name__ == '__main__':

    data_root_path = Path('/root/data/raw/12152021_emotion3D_bw')
    folders = [item for item in data_root_path.iterdir() if item.is_dir()]
    for folder in folders:
        data_preprocessor = Emotion3DPreprocessor(
            source_path=str(folder),
            source_annotation_folder='gt_jsons',
            source_image_folder='images',
            destination_path='/root/data/processed/synthetic_cabin_bw',
            annotation_folder='annotations',
            image_folder='images',
            train_image_folder='train',
            val_image_folder='val',
            test_image_folder='test',
            person_detection_folder='person_detection_results',
            keypoint_detection_folder='keypoint_detection_results',
            visualization_folder='visualizations',
            source_height=1024,
            source_width=1280,
            destination_height=1024,
            destination_width=1280,
            val_subset={'Amit', 'Claudia'},
            test_subset={'Alison', 'Ryo'}
        )

        data_preprocessor.format_dataset(
            annotation_template,
            num_train=1000,
            num_val=1000,
            num_test=1000
        )
    # print(gather_camera_position())
