import json
import pandas as pd
import numpy as np


class ExperimentSummerizer:
    def __init__(self, experiment_saved_path, experiment_labels):
        self.saved_model_path = experiment_saved_path 
        self.experiment_labels = experiment_labels
        self.test_mpjpe_list = []
        self.test_pjpe_list = []
        self.test_activity_mpjpe_list = []
        self.test_activity_macro_mpjpe_list = []
        self.avg_mpjpe = None
        self.avg_activity_macro_mpjpe = None
        self.avg_pjpe = None
        self.avg_activity_mpjpe = None

    def add_result(self, test_mpjpe, test_pjpe, test_activity_mpjpe, test_activity_macro_mpjpe):
        self.test_mpjpe_list.append(test_mpjpe)
        self.test_pjpe_list.append(test_pjpe)
        self.test_activity_mpjpe_list.append(test_activity_mpjpe)
        self.test_activity_macro_mpjpe_list.append(test_activity_macro_mpjpe)

    def calculate(self):
        self.avg_mpjpe = np.mean(self.test_mpjpe_list)
        self.avg_pjpe = pd.DataFrame(self.test_pjpe_list, index=self.experiment_labels).mean()
        self.avg_activity_mpjpe = pd.DataFrame(self.test_activity_mpjpe_list, index=self.experiment_labels).mean()
        self.avg_activity_macro_mpjpe = np.mean(self.test_activity_macro_mpjpe_list)
        with open(f'{self.saved_model_path}/summarize.json', 'w') as f:
            result = json.dumps(dict(
                avg_mpjpe=self.avg_mpjpe,
                avg_pjpe=self.avg_pjpe.to_dict(),
                avg_activity_macro_mpjpe=self.avg_activity_macro_mpjpe,
                avg_activity_mpjpe=self.avg_activity_mpjpe.to_dict()
            ), indent=4)
            f.write(result)

    def print_raw_result(self):
        print(f'Test MPJPE:\n{self.test_mpjpe_list}\n')
        print(f'Test PJPE:\n{pd.DataFrame(self.test_pjpe_list, index=self.experiment_labels)}\n')
        print(f'Test Macro Average Activity-MPJPE:\n{self.test_activity_macro_mpjpe_list}\n')
        print(f'Test Activity MPJPE:\n{pd.DataFrame(self.test_activity_mpjpe_list, index=self.experiment_labels)}')

    def print_summarize_result(self):
        print(f'MPJPE = {self.avg_mpjpe}\n')
        print(f'PJPE =\n{self.avg_pjpe}\n')
        print(f'Macro Average Activity-MPJPE = {self.avg_activity_macro_mpjpe}\n')
        print(f'Activity-MPJPE =\n{self.avg_activity_mpjpe}')
