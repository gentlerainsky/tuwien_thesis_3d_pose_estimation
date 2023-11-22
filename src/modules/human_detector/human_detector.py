from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import inference_detector, init_detector
# from mmdet.registry import VISUALIZERS
from mmengine.visualization import Visualizer
import mmcv
import matplotlib.pyplot as plt
import os
import torch
from pathlib import Path


class HumanDetector:
    def __init__(
        self,
        config_path,
        pretrained_path,
        checkpoint_path,
        data_root_path,
        device='cpu',
        working_directory='./human_detector_wd',
        log_level='INFO'
    ):
        self.config_path = config_path
        self.pretrained_path = pretrained_path
        self.checkpoint_path = checkpoint_path
        self.data_root_path = Path(data_root_path)
        self.device = device
        self.working_directory = working_directory
        self.log_level = log_level
        self.load_from_checkpoint = False

    def update_config(self):
        self.config = Config.fromfile(self.config_path)
        self.config.data_root = self.data_root_path.as_posix()
        if self.load_from_checkpoint:
            self.config.load_from = self.checkpoint_path
        else:
            self.config.load_from = self.pretrained_path
        self.config.work_dir = self.working_directory
        self.config.log_level = self.log_level

        # Training
        # self.config.train_cfg['by_epoch'] = True
        self.config.train_cfg['max_epochs'] = 1
        self.config.default_hooks['logger']['interval'] = 500
        dataset_type = "CocoDataset"
        self.config.train_dataloader['dataset']['type'] = dataset_type
        self.config.train_dataloader['dataset']['data_root'] = self.data_root_path.as_posix()
        self.config.train_dataloader['dataset']['ann_file'] = "annotations/person_keypoints_train.json"
        self.config.train_dataloader['dataset']['data_prefix']['img'] = 'images/train/'
        self.config.train_dataloader['dataset']['filter_cfg']['filter_empty_gt'] = True
        self.config.train_dataloader['dataset']['filter_cfg']['min_size'] = 32

        # Validation
        self.config.val_dataloader['dataset']['type'] = dataset_type
        self.config.val_dataloader['dataset']['data_root'] = self.data_root_path.as_posix()
        self.config.val_dataloader['dataset']['ann_file'] = "annotations/person_keypoints_val.json"
        self.config.val_dataloader['dataset']['data_prefix']['img'] = 'images/val/'
        self.config.val_dataloader['dataset']['test_mode'] = True
        self.config.val_evaluator['ann_file'] = (self.data_root_path / "annotations/person_keypoints_val.json").as_posix()

        # # Testing
        self.config.test_dataloader['dataset']['type'] = dataset_type
        self.config.test_dataloader['dataset']['data_root'] = self.data_root_path.as_posix()
        self.config.test_dataloader['dataset']['ann_file'] = "annotations/person_keypoints_test.json"
        self.config.test_dataloader['dataset']['data_prefix']['img'] = 'images/test/'
        self.config.test_dataloader['dataset']['test_mode'] = True
        self.config.test_evaluator['ann_file'] = (self.data_root_path / "annotations/person_keypoints_test.json").as_posix()

        self.config.visualizer['vis_backends'] = [
            dict(type='LocalVisBackend'),
            dict(type='TensorboardVisBackend'),
            dict(type='WandbVisBackend'),
        ]

        self.runner = Runner.from_cfg(self.config)
        if self.load_from_checkpoint:
            self.runner.load_or_resume()
        self.model = self.runner.model
        self.model.cfg = self.runner.cfg

    def load_pretrained(self):
        self.load_from_checkpoint = True
        self.update_config()

    def finetune(self):
        self.update_config()
        self.model.train()
        self.runner.train()
        with open(os.path.join(self.config.work_dir, 'last_checkpoint')) as f:
            filepath = f.readline()
            print('new_checkpoint_path', filepath)
            self.checkpoint_path = filepath

    def inference(self, img_path):
        self.model.eval()
        img = mmcv.imread(img_path)
        result = inference_detector(self.model, img)
        return result
    
    def get_bbox(self, img_path):
        result = self.inference(img_path)
        pred = result.pred_instances
        human_mask = (pred['labels'] == 0)
        bboxes = pred['bboxes'][human_mask]
        scores = pred['scores'][human_mask]
        if pred['scores'][human_mask].shape[0] > 0:
            index = pred['scores'][human_mask].argmax()
            if pred['scores'][human_mask][index] < 0.5:
                print(f'{img_path} has mask with {pred["scores"][human_mask][index]}')
            index = index.unsqueeze(0)
            bboxes = pred['bboxes'][human_mask][index]
            scores = pred['scores'][human_mask][index]
        return {
            'bboxes': bboxes,
            'scores': scores,
        }

    def visualise(self, img_path, bbox):
        img = mmcv.imread(img_path)
        img = mmcv.imconvert(img, 'bgr', 'rgb')
        if bbox is None:
            result = self.get_bbox(img_path)
            bbox = result['bboxes']
        visualizer = Visualizer(image=img)
        visualizer.draw_bboxes(bbox)
        imgplot = plt.imshow(visualizer.get_image())
        plt.show()

    def test(self):
        self.runner.test()
