import torch
from modules.lifter_2d_3d.model.common.lit_base_model import LitBaseModel
from modules.lifter_2d_3d.model.linear_model.network.linear_model import BaselineModel


class LitSimpleBaselineLinear(LitBaseModel):
    def __init__(
        self,
        **args
    ):
        super().__init__(**args)
        self.model = BaselineModel(
            exclude_ankle=self.exclude_ankle,
            exclude_knee=self.exclude_knee
        )

    def preprocess_input(self, x, y, valid, activity):
        x = torch.flatten(x, start_dim=1).float()
        y = torch.flatten(y, start_dim=1).float()
        valid = torch.flatten(valid, start_dim=1)
        return x, y, valid, activity
