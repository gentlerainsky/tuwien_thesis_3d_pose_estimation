import os
cwd = os.getcwd()
import sys
sys.path.append(cwd)

import pytorch_lightning as pl

from modules.experiments.experiment import Experiment
from modules.experiments.timer import Timer
from experiments.experiment_config import (
    ALL_LIGHTNING_MODELS,
    SYNTHETIC_CABIN_VIEWPOINTS,
    get_synthetic_cabin_ir_loaders
)

pl.seed_everything(1234)

timer = Timer()
timer.start()
for viewpoint in SYNTHETIC_CABIN_VIEWPOINTS:
    constructed_loader = get_synthetic_cabin_ir_loaders(viewpoint)
    for LitModel in ALL_LIGHTNING_MODELS:
        print(f'RUNNING FOR MODEL: {LitModel.__name__} / VIEWPOINT: {viewpoint}')
        experiment = Experiment(
            LitModel=LitModel,
            constructed_loader=constructed_loader,
            saved_model_path=f'saved_lifter_2d_3d_model/rq1/{LitModel.__name__}/{viewpoint}',
            model_parameters=dict(
                exclude_ankle=True,
                exclude_knee=True
            )
        )
        experiment.setup(
            trainer_config=dict(max_epoch=5)
        )
        experiment.train()
        experiment.test()
        experiment.print_result()
        timer.lap()
        print(timer)
    timer.finish()
    print(timer)
