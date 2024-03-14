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
    four_actors_samples,
)

pl.seed_everything(1234)

subset_setup = {
    # "all_actors": [all_train_actors],
    # "single_actor": all_train_actors,
    "single_actor": ["vp1"],
    # "two_actors": two_actors_samples,
    # "four_actors": four_actors_samples,
}

subset_percentages = [1, 2, 5, 10, 25, 50, 100]
# subset_percentages = [100]
# subset_percentages = [1]
timer = Timer()
timer.start()

for setup_name in subset_setup.keys():
    for subset_percentage in subset_percentages:
        for viewpoint in DRIVE_AND_ACT_VIEWPOINTS:
            constructed_loader = get_drive_and_act_loaders(
                viewpoint, subset_percentage=subset_percentage
            )
            for LitModel in ALL_LIGHTNING_MODELS:
                saved_model_path_root = f"saved_lifter_2d_3d_model/rq2/{LitModel.__name__}/drive_and_act/{viewpoint}/{setup_name}/subset_{subset_percentage}"
                summerizer = ExperimentSummarizer(
                    experiment_saved_path=saved_model_path_root, experiment_labels=None
                )
                labels = []
                for subset in subset_setup[setup_name]:
                    if setup_name == "all_actors":
                        subset_name = "all_actors"
                    elif setup_name != "single_actor":
                        subset_name = "_".join(sorted(subset))
                    else:
                        subset_name = subset
                    labels.append(subset_name)
                    print(
                        f"RUNNING FOR MODEL: {LitModel.__name__} / VIEWPOINT: {viewpoint} "
                        + f"/ SUBSET: {setup_name} / SAMPLE: {subset_name}"
                    )
                    saved_model_path = f"{saved_model_path_root}/{subset_name}"
                    experiment = Experiment(
                        LitModel=LitModel,
                        constructed_loader=constructed_loader,
                        saved_model_path=saved_model_path,
                        model_parameters=dict(exclude_ankle=True, exclude_knee=True),
                    )
                    experiment.setup()
                    experiment.train()
                    experiment.test()
                    # experiment.remove_saved_model()
                    experiment.print_result()
                    summerizer.add_result(
                        test_mpjpe=experiment.test_mpjpe,
                        test_pjpe=experiment.test_pjpe[0],
                        test_activity_mpjpe=experiment.test_activity_mpjpe[0],
                        test_activity_macro_mpjpe=experiment.test_activity_macro_mpjpe,
                        test_p_mpjpe=experiment.test_p_mpjpe,
                        test_p_pjpe=experiment.test_p_pjpe[0],
                        test_p_activity_mpjpe=experiment.test_p_activity_mpjpe[0],
                        test_p_activity_macro_mpjpe=experiment.test_p_activity_macro_mpjpe,
                    )
                    timer.lap()
                    print(timer)
                summerizer.experiment_labels = labels
                summerizer.calculate()
                summerizer.print_summarize_result()
        timer.finish()
        print(timer)
