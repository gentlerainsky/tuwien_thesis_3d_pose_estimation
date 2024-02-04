import os
cwd = os.getcwd()
import sys
sys.path.append(cwd)

import pytorch_lightning as pl

from modules.experiments.experiment import Experiment
from modules.experiments.experiment_summarizer import ExperimentSummarizer
from modules.experiments.timer import Timer
from modules.experiments.dataset import all_train_actors
from experiments.experiment_config import (
    ALL_LIGHTNING_MODELS,
    DRIVE_AND_ACT_VIEWPOINTS,
    get_drive_and_act_loaders,
    two_actors_samples,
    four_actors_samples
)

pl.seed_everything(1234)

subset_setup = {
    'single_actor': all_train_actors,
    'two_actors': two_actors_samples,
    'four_actors': four_actors_samples,
    'all_actors': [all_train_actors]
}

timer = Timer()
timer.start()
for viewpoint in DRIVE_AND_ACT_VIEWPOINTS:
    constructed_loader = get_drive_and_act_loaders(viewpoint)
    for LitModel in ALL_LIGHTNING_MODELS:
        for setup_name in subset_setup.keys():
            saved_model_path_root = f'saved_lifter_model/rq2/{LitModel.__name__}/drive_and_act/{viewpoint}/{setup_name}'
            summerizer = ExperimentSummarizer(
                experiment_saved_path=saved_model_path_root,
                experiment_labels=subset_setup.keys()
            )
            for subset in subset_setup[setup_name]:
                if setup_name == 'all_actors':
                    subset_name = 'all_actors'
                elif setup_name != 'single_actor':
                    subset_name = '_'.join(sorted(subset))
                else:
                    subset_name = subset
                print(f'RUNNING FOR MODEL: {LitModel.__name__} / VIEWPOINT: {viewpoint} '
                      + '/ SUBSET: {subset} / SAMPLE: {subset_name}')
                experiment = Experiment(
                    LitModel=LitModel,
                    constructed_loader=constructed_loader,
                    saved_model_path=\
                        f'{saved_model_path_root}/{subset_name}',
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
                summerizer.add_result(
                    test_mpjpe=experiment.test_mpjpe,
                    test_pjpe=experiment.test_pjpe[0],
                    test_activity_mpjpe=experiment.test_activity_mpjpe[0],
                    test_activity_macro_mpjpe=experiment.test_activity_macro_mpjpe
                )
                timer.lap()
                print(timer)
            summerizer.calculate()
            summerizer.print_summarize_result()
    timer.finish()
    print(timer)
