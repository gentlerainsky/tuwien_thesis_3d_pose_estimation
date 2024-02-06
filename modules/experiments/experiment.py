import os
import json
import shutil
from pathlib import Path
import pandas as pd
from modules.experiments.trainer import create_trainer
import logging


class Experiment:
    def __init__(
            self,
            LitModel,
            constructed_loader,
            saved_model_path,
            pretrained_model_path=None,
            enable_log=True,
            enable_progress_bar=True,
            model_parameters=None
        ):
        self.LitModel = LitModel
        self.saved_model_path = saved_model_path
        self.pretrained_model_path = pretrained_model_path
        self.constructed_loader = constructed_loader
        self.enable_log = enable_log
        self.enable_progress_bar = enable_progress_bar
        self.model_parameters = model_parameters
        self.all_activities = None
        self.trainer = None
        self.model_checkpoint_callback = None
        self.best_checkpoint_path = None
        self.setup_dataset()
        self.setup_model()
        self.setup_trainer()

    def setup_dataset(self):
        constructed_dataset = self.constructed_loader
        self.train_loader = constructed_dataset['train_loader']
        self.val_loader = constructed_dataset['val_loader']
        self.test_loader = constructed_dataset['test_loader']
        self.all_activities = constructed_dataset['all_activities']
        if self.enable_log:
            print(
                'train_dataset', len(self.train_loader.dataset),
                'val_dataset', len(self.val_loader.dataset),
                'test_dataset', len(self.test_loader.dataset)
            )
    
    def remove_saved_model(self):
        # to make sure that it doesn't remove anything outside the project
        root = Path(os.getcwd())
        child = Path(self.saved_model_path).resolve()
        if root in child.parents:
            shutil.rmtree(child)

    def create_log_folder(self):
        if not os.path.exists(self.saved_model_path):
            os.makedirs(self.saved_model_path)

    def _load_pretrained(self):
        with open(f'{self.pretrained_model_path}/best_model_path.txt', 'r') as f:
            pretrained_checkpoint_path = f.readline()
        self.lit_model = self.LitModel.load_from_checkpoint(pretrained_checkpoint_path)
        self.lit_model.set_all_activities(self.all_activities)

    def setup_model(self):
        if self.pretrained_model_path is not None:
            if self.enable_log:
                print(f'loaded from {self.pretrained_model_path}')
            self._load_pretrained()
        else:
            if self.enable_log:
                print(f'create new {self.LitModel.__name__} model')
            self.lit_model = self.LitModel(
                **self.model_parameters,
                all_activities=self.all_activities
            )

    def setup_trainer(self, trainer_config=None):
        if trainer_config is None:
            trainer_config = {}
        args = dict(
            # run full sanity check on validation set
            num_sanity_val_steps=-1,
            saved_model_path=self.saved_model_path,
            enable_progress_bar=self.enable_progress_bar,
            **trainer_config
        )
        output = create_trainer(**args)
        self.trainer = output['trainer']
        self.model_checkpoint_callback = output['model_checkpoint_callback']

    def setup(self, trainer_config=None):
        if self.enable_log is False:
            logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
            logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
        self.remove_saved_model()
        self.create_log_folder()
        self.setup_dataset()
        self.setup_model()
        self.setup_trainer(trainer_config)

    def train(self):
        self.trainer.fit(self.lit_model, self.train_loader, self.val_loader)
        with open(f'{self.saved_model_path}/best_model_path.txt', 'w') as f:
            f.writelines(self.model_checkpoint_callback.best_model_path)
        self.best_checkpoint_path = self.model_checkpoint_callback.best_model_path
        self._write_training_log()

    def print_result(self):
        print(f'MPJPE = {self.test_mpjpe}')
        print(f'PJPE =\n{pd.DataFrame(self.test_pjpe, index=["PJPE"]).T}')
        if self.test_activity_macro_mpjpe is not None:
            print(f'Activity-based Macro Average MPJPE = {self.test_activity_macro_mpjpe}')
        if self.test_activity_mpjpe is not None:
            print(f'Activity-base MPJPE =\n{pd.DataFrame(self.test_activity_mpjpe, index=["MPJPE"]).T}')
        # Procrusted Version
        print(f'P-MPJPE = {self.test_p_mpjpe}')
        print(f'P-PJPE =\n{pd.DataFrame(self.test_p_pjpe, index=["P-PJPE"]).T}')
        if self.test_p_activity_macro_mpjpe is not None:
            print(f'Activity-based Macro Average P-MPJPE = {self.test_p_activity_macro_mpjpe}')
        if self.test_p_activity_mpjpe is not None:
            print(f'Activity-base P-MPJPE =\n{pd.DataFrame(self.test_p_activity_mpjpe, index=["P-MPJPE"]).T}')

    def _write_training_log(self):
        with open(f'{self.saved_model_path}/training_log.json', 'w') as f:
            info = dict(
                train_history=self.lit_model.train_loss_log,
                val_history=self.lit_model.val_history
            )
            f.write(json.dumps(info, indent=2))

    def _write_test_results(self):
        with open(f'{self.saved_model_path}/test_result.json', 'w') as f:
            info = dict(
                checkpoint_path=self.best_checkpoint_path,
                mpjpe=self.test_mpjpe,
                pjpe=self.test_pjpe,
                activity_mpjpe=self.test_activity_mpjpe,
                activity_macro_mpjpe=self.test_activity_macro_mpjpe,
                # Procrusted Version
                p_mpjpe=self.test_p_mpjpe,
                p_pjpe=self.test_p_pjpe,
                p_activity_mpjpe=self.test_p_activity_mpjpe,
                p_activity_macro_mpjpe=self.test_p_activity_macro_mpjpe
            )
            f.write(json.dumps(info, indent=2))

    def test(self):
        if self.best_checkpoint_path is None:
            self.trainer.test(
                self.lit_model,
                dataloaders=self.test_loader
            )
        else:
            self.trainer.test(
                ckpt_path=self.best_checkpoint_path,
                dataloaders=self.test_loader
            )
        self.test_mpjpe = self.trainer.model.test_history[0]['mpjpe']
        self.test_pjpe = self.trainer.model.test_history[0]['pjpe']
        self.test_activity_mpjpe = pd.DataFrame(
            self.trainer.model.test_history[0]['activities_mpjpe'], index=['mpjpe']
        ).to_dict(orient='records')
        self.test_activity_macro_mpjpe = self.trainer.model.test_history[0]['activity_macro_mpjpe']
        # Procrusted Version
        self.test_p_mpjpe = self.trainer.model.test_history[0]['p_mpjpe']
        self.test_p_pjpe = self.trainer.model.test_history[0]['p_pjpe']
        self.test_p_activity_mpjpe = pd.DataFrame(
            self.trainer.model.test_history[0]['p_activities_mpjpe'], index=['p_mpjpe']
        ).to_dict(orient='records')
        self.test_p_activity_macro_mpjpe = self.trainer.model.test_history[0]['p_activity_macro_mpjpe']
        self._write_test_results()
