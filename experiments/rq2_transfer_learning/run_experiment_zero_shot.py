import os
cwd = os.getcwd()
import sys
sys.path.append(cwd)

import pytorch_lightning as pl

from modules.experiments.experiment import Experiment
from modules.experiments.timer import Timer
from experiments.experiment_config import (
    ALL_LIGHTNING_MODELS,
    get_drive_and_act_loaders,
    synthetic_cabin_ir_1m_preprocess_loaders,
    driver_and_act_pretrained_map
)

pl.seed_everything(1234)

timer = Timer()
timer.start()
for viewpoint in driver_and_act_pretrained_map.keys():
    print(f'Start Loading {viewpoint} Dataloader')
    constructed_loader = get_drive_and_act_loaders(viewpoint)
    print(f'Finish Loading {viewpoint} Dataloader')
    for LitModel in ALL_LIGHTNING_MODELS:
        print(f'RUNNING FOR MODEL: {LitModel.__name__} / VIEWPOINT: {viewpoint}')
        pretrained_model_path = f'saved_lifter_2d_3d_model/rq2/{LitModel.__name__}/synthetic_cabin_ir_1m/{driver_and_act_pretrained_map[viewpoint]}'
        experiment = Experiment(
            LitModel=LitModel,
            constructed_loader=constructed_loader,
            pretrained_model_path=pretrained_model_path,
            saved_model_path=f'saved_lifter_2d_3d_model/rq2/{LitModel.__name__}/zero_shot/{viewpoint}_with_all',
            model_parameters=dict(
                exclude_ankle=True,
                exclude_knee=True
            )
        )
        experiment.setup()
        # experiment.train()
        experiment.test()
        timer.lap()
        experiment.print_result()
        print(timer)
    timer.finish()
    print('Finish Experiments')
    print(timer)
