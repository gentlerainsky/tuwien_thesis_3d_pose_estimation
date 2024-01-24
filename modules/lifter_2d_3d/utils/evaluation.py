import numpy as np
import pandas as pd
from modules.data_preprocessing.definition import coco_keypoint_names


class Evaluator:
    def __init__(self, all_activities=None):
        self.all_activities = all_activities
        if all_activities is not None:
            self.all_activities = list(all_activities)
        self.pjpe = []
        self.mpjpe = []
        self.activities_mpjpe = {}
        self.activity_macro_mpjpe = []

    def reset(self):
        self.pjpe = []
        self.mpjpe = []
        self.activities_mpjpe = {}

    def calculate_mpjpe(self, pred_3d, gt_3d, valid):
        pjpe_list = []
        mpjpe_list = []
        pred_3d = pred_3d.reshape(pred_3d.shape[0], -1, 3)
        gt_3d = gt_3d.reshape(gt_3d.shape[0], -1, 3)
        # Loop over each sample in a batch
        for i in range(gt_3d.shape[0]):
            mask = (gt_3d[i] != 0)
            mask = np.tile(valid[i].reshape([-1, 1]), (1, 3))
            pjpe = np.sqrt(
                np.power((pred_3d[i] - gt_3d[i]), 2).sum(axis=1, where=mask)
            )
            pjpe_list.append(pjpe)
            mask = (pjpe != 0)
            mpjpe = np.mean(pjpe, axis=0, where=mask)
            mpjpe_list.append(mpjpe)
        return pjpe_list, mpjpe_list

    def add_result(self, pred_3d, gt_3d, valid, input_activities=None):
        pjpe_list, mpjpe_list = self.calculate_mpjpe(pred_3d, gt_3d, valid)
        self.pjpe += pjpe_list
        self.mpjpe += mpjpe_list
        # calculate action-based mpjpe
        if self.all_activities is not None:
            input_activities = np.array(input_activities)
            activities_mpjpe = []
            for activity in self.all_activities:
                mask = np.array(input_activities == activity)
                if np.all(~mask):
                    continue
                _, mpjpe = self.calculate_mpjpe(pred_3d[mask], gt_3d[mask], valid)
                if activity not in self.activities_mpjpe:
                    self.activities_mpjpe[activity] = []
                self.activities_mpjpe[activity] += mpjpe
                activities_mpjpe += mpjpe
            # macro average of activities mpjpe
            if len(activities_mpjpe) > 0:
                self.activity_macro_mpjpe += [np.mean(activities_mpjpe)]

    def get_result(self):
        mpjpe = np.nanmean(np.array(self.mpjpe)) * 1000
        pjpe_data = np.array(self.pjpe).mean(axis=0) * 1000
        pjpe = pd.DataFrame(
            data=pjpe_data,
            index=coco_keypoint_names[:pjpe_data.shape[0]],
            columns=['PJPE']
        ).T.to_dict(orient='records')
        activity_macro_mpjpe = None
        if len(self.activity_macro_mpjpe) > 0:
            activity_macro_mpjpe = np.mean(self.activity_macro_mpjpe) * 1000
        activities_mpjpe_data = []
        activities_mpjpe = None
        if self.all_activities is not None:
            activities = list(self.activities_mpjpe.keys())
            for activity in activities:
                activities_mpjpe_data.append(
                    np.array(self.activities_mpjpe[activity]).mean() * 1000
                )
            activities_mpjpe = pd.DataFrame(
                data=activities_mpjpe_data,
                index=activities,
                columns=['MPJPE']
            ).T.to_dict(orient='records')
        return pjpe, mpjpe, activities_mpjpe, activity_macro_mpjpe
