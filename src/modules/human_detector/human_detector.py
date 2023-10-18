from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.apis import inference_detector, init_detector
# from mmdet.registry import VISUALIZERS
from mmengine.visualization import Visualizer
import mmcv
import matplotlib.pyplot as plt
import os
import torch


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
        self.data_root_path = data_root_path
        self.device = device
        self.working_directory = working_directory
        self.log_level = log_level
        self.load_from_checkpoint = False

    def update_config(self):
        self.config = Config.fromfile(self.config_path)
        self.config.data_root = self.data_root_path
        if self.load_from_checkpoint:
            self.config.load_from = self.checkpoint_path
        else:
            self.config.load_from = self.pretrained_path
        self.config.work_dir = self.working_directory
        self.config.log_level = self.log_level
        self.runner = Runner.from_cfg(self.config)
        if self.load_from_checkpoint:
            self.runner.load_or_resume()
        self.model = self.runner.model
        self.model.cfg = self.runner.cfg

    # def load_pretrained(self):
    #     self.load_from_checkpoint = True
    #     self.update_config()

    def load_pretrained(self):
        self.load_from_checkpoint = True
        self.update_config()
        # self.model = init_detector(
        #     self.config_path,
        #     self.checkpoint_path,
        #     device=self.device
        # )

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
        # mask = (result.pred_instances['labels'] == 0) & (result.pred_instances['scores'] > 0.7)
        # human_mask = (pred['labels'] == 0).to(torch.long) 
        # index = pred['scores'][human_mask].argmax()
        # if pred['scores'][index] < 0.5:
        #     print(f'{img_path} has mask with {pred["scores"][index]}')
        # index = index.unsqueeze(0)
        # bboxes = pred['bboxes'][index]

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
        # visualizer = VISUALIZERS.build(self.runner.cfg.visualizer)
        # visualizer.add_datasample(
        #     'result',
        #     img,
        #     data_sample=result,
        #     draw_gt=False,
        #     show=True)
        # # img = mpimg.imread('your_image.png')
        # imgplot = plt.imshow(visualizer.get_image())
        # plt.show()
        visualizer = Visualizer(image=img)
        # single bbox formatted as [xyxy]
        visualizer.draw_bboxes(bbox)
        # draw multiple bboxes
        # visualizer.show()
        imgplot = plt.imshow(visualizer.get_image())
        plt.show()

    def test(self):
        # self.update_config()
        self.runner.test()

# class HumanDetector:
#     def __init__(
#         self,
#         config_path,
#         pretrained_path,
#         checkpoint_path,
#         data_root_path,
#         device='cpu',
#         working_directory='./human_detector_wd',
#         log_level='INFO'
#     ):
#         self.config_path = config_path
#         self.pretrained_path = pretrained_path
#         self.checkpoint_path = checkpoint_path
#         self.data_root_path = data_root_path
#         self.device = device
#         self.config = Config.fromfile(self.config_path)
#         self.config.data_root = data_root_path
#         self.config.load_from = self.pretrained_path
#         self.config.work_dir = working_directory
#         self.config.log_level = log_level
#         self.runner = Runner.from_cfg(self.config)
#         self.model = self.runner.model
#         self.model.cfg = self.runner.cfg

#     def load_pretrained(self):
#         self.model = init_detector(
#             self.config_path,
#             self.checkpoint_path,
#             device=self.device
#         )

#     def finetune(self):
#         self.model.train()
#         self.runner.train()
#         with open(os.path.join(self.config.work_dir, 'last_checkpoint')) as f:
#             filepath = f.readline()
#             print('new_checkpoint_path', filepath)
#             self.checkpoint_path = filepath

#     def inference(self, img_path):
#         self.model.eval()
#         img = mmcv.imread(img_path)
#         result = inference_detector(self.model, img)
#         return result
    
#     def get_bbox(self, img_path):
#         result = self.inference(img_path)
#         return {
#             'bboxes': result.pred_instances['bboxes'],
#             'scores': result.pred_instances['scores']
#         }

#     def visualise(self, img_path):
#         img = mmcv.imread(img_path)
#         img = mmcv.imconvert(img, 'bgr', 'rgb')
#         result = self.inference(img_path)
#         visualizer = VISUALIZERS.build(self.runner.cfg.visualizer)
#         visualizer.add_datasample(
#             'result',
#             img,
#             data_sample=result,
#             draw_gt=False,
#             show=True)
#         # img = mpimg.imread('your_image.png')
#         imgplot = plt.imshow(visualizer.get_image())
#         plt.show()


if __name__ == '__main__':
    pass
