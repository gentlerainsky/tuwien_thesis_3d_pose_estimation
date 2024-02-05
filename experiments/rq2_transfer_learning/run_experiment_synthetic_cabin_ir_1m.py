import os
cwd = os.getcwd()
import sys
sys.path.append(cwd)

import pytorch_lightning as pl

from modules.experiments.experiment import Experiment
from modules.experiments.timer import Timer
from experiments.experiment_config import (
    ALL_LIGHTNING_MODELS,
    get_processed_synthetic_cabin_ir_1m_loaders,
    synthetic_cabin_ir_1m_preprocess_loaders
)

pl.seed_everything(1234)

timer = Timer()
timer.start()
for viewpoint in synthetic_cabin_ir_1m_preprocess_loaders.keys():
    print(f'Start Loading {viewpoint} Dataloader')
    constructed_loader = get_processed_synthetic_cabin_ir_1m_loaders(viewpoint)
    print(f'Finish Loading {viewpoint} Dataloader')
    for LitModel in ALL_LIGHTNING_MODELS:
        print(f'RUNNING FOR MODEL: {LitModel.__name__} / VIEWPOINT: {viewpoint}')
        experiment = Experiment(
            LitModel=LitModel,
            constructed_loader=constructed_loader,
            saved_model_path=f'saved_lifter_2d_3d_model/rq2/{LitModel.__name__}/synthetic_cabin_ir_1m/{viewpoint}',
            model_parameters=dict(
                exclude_ankle=True,
                exclude_knee=True
            )
        )
        experiment.setup(
            trainer_config=dict(
                # max_epoch=1,
                # val_check_period=1,
                # early_stopping_patience=1,
            )
        )
        experiment.train()
        experiment.test()
        timer.lap()
        experiment.print_result()
        print(timer)
    timer.finish()
    print('Finish Experiments')
    print(timer)
