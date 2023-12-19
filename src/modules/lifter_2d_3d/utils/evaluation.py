import numpy as np
import pandas as pd
from src.modules.data_preprocessing.definition import coco_keypoint_names


class Evaluator:
    def __init__(self, all_activities=None):
        self.all_activities = all_activities
        if all_activities is not None:
            self.all_activities = list(all_activities)
        self.pjpe = []
        self.mpjpe = []
        self.activities_mpjpe = {}

    def reset(self):
        self.pjpe = []
        self.mpjpe = []
        self.activities_mpjpe = {}

    def calculate_mpjpe(self, pred_3d, gt_3d):
        pjpe_list = []
        mpjpe_list = []
        pred_3d = pred_3d.reshape(pred_3d.shape[0], -1, 3)
        gt_3d = gt_3d.reshape(gt_3d.shape[0], -1, 3)
        # Loop over each sample in a batch
        for i in range(gt_3d.shape[0]):
            mask = (gt_3d[i] != 0)
            pjpe = np.sqrt(
                np.power((pred_3d[i] - gt_3d[i]), 2).sum(axis=1, where=mask)
            )
            pjpe_list.append(pjpe)
            mask = (pjpe != 0)
            mpjpe = np.mean(pjpe, axis=0, where=mask)
            mpjpe_list.append(mpjpe)
        return pjpe_list, mpjpe_list

    def add_result(self, pred_3d, gt_3d, input_activities=None):
        pjpe_list, mpjpe_list = self.calculate_mpjpe(pred_3d, gt_3d)
        self.pjpe += pjpe_list
        self.mpjpe += mpjpe_list
        # calculate action-based mpjpe
        if self.all_activities is not None:
            input_activities = np.array(input_activities)
            for activity in self.all_activities:
                mask = np.array(input_activities == activity)
                if np.all(~mask):
                    continue
                _, activities_mpjpe = self.calculate_mpjpe(pred_3d[mask], gt_3d[mask])
                if activity not in self.activities_mpjpe:
                    self.activities_mpjpe[activity] = []
                self.activities_mpjpe[activity] += activities_mpjpe

    def get_result(self):
        mpjpe = np.nanmean(np.array(self.mpjpe)) * 1000
        pjpe_data = np.array(self.pjpe).mean(axis=0) * 1000
        pjpe = pd.DataFrame(
            data=pjpe_data,
            index=coco_keypoint_names[:pjpe_data.shape[0]],
            columns=['PJPE']
        )
        activities_mpjpe = {}
        if self.all_activities is not None:
            for activity in self.activities_mpjpe.keys():
                activities_mpjpe[activity] = np.array(self.activities_mpjpe[activity]).mean() * 1000
        return pjpe, mpjpe, activities_mpjpe