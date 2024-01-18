import os
import json
import shutil
from pathlib import Path
import pandas as pd
from modules.experiments.trainer import create_trainer
from modules.utils.convention import get_saved_model_path


class Experiment:
    def __init__(
            self,
            LitModel,
            dataset_constructor,
            saved_model_path,
            pretrained_model_path,
            enable_log=True,
            enable_progress_bar=True
        ):
        self.LitModel = LitModel
        self.saved_model_path = saved_model_path
        self.pretrain_model_path = pretrained_model_path
        self.dataset_constructor = dataset_constructor
        self.enable_log = enable_log
        self.enable_progress_bar = enable_progress_bar
        self.all_activities = None
        self.saved_model_path = None
        self.checkpoint_path = None
        self.trainer = None
        self.model_checkpoint = None
        self.best_checkpoint_path = None
        self.setup_dataset()
        self.setup_model()
        self.setup_trainer()

    def setup_dataset(self):
        constructed_dataset = self.dataset_constructor()
        self.train_loader = constructed_dataset['train_loader']
        self.val_loader = constructed_dataset['val_loader']
        self.checkpoint_pathtest_loader = constructed_dataset['test_loader']
        self.all_activities = constructed_dataset['all_activities']
        if self.enable_log:
            print(
                'train_dataset', len(self.train_loader.train_dataset),
                'val_dataset', len(self.val_loader.val_dataset),
                'test_dataset', len(self.test_loader.test_dataset)
            )
    
    def remove_saved_model(self):
        # to make sure that it doesn't remove anything outside the project
        root = Path(os.getcwd())
        child = Path(self.saved_model_path)
        if root in child.parents:
            shutil.rmtree(self.saved_model_path)

    def _load_pretrained(self):
        with open(f'{self.pretrain_model_path}/best_model_path.txt', 'r') as f:
            pretrained_checkpoint_path = f.readline()
        self.lit_model = self.LitModel.load_from_checkpoint(pretrained_checkpoint_path)
        self.lit_model.set_all_activities(self.all_activities)

    def setup_model(self):
        if self.pretrain_model_path is not None:
            self._load_pretrained()
        else:
            self.lit_model = self.LitModel()

    def setup_trainer(self):
        self.saved_model_path = get_saved_model_path(
            model_name='linear_model',
            model_suffix='all_actors',
            trained_dataset_name='drive_and_act',
            trained_datasubset_name='predicted_2d_all_actors',
            pretrained_dataset_name='synthetic_cabin_ir',
            pretrained_datasubset_name='A_Pillar_Codriver',
        )
        output = create_trainer(
            self.saved_model_path,
            enable_progress_bar=self.enable_progress_bar
        )
        self.trainer = output['trainer']
        self.model_checkpoint = output['model_checkpoint']

    def train(self):
        self.trainer.fit(self.lit_model, self.train_loader, self.val_loader)
        with open(f'{self.saved_model_path}/best_model_path.txt', 'w') as f:
            f.writelines(self.model_checkpoint.best_model_path)
        self.best_checkpoint_path = self.model_checkpoint.best_model_path

    def _write_test_results(self):
        with open(f'{self.saved_model_path}/test_results') as f:
            info = {
                'checkpoint_path': self.best_checkpoint_path,
                'mpjpe': self.trainer.model.test_history[0]['mpjpe'],
                'pjpe': self.trainer.model.test_history[0]['pjpe'],
                'activities_mpjpe': pd.DataFrame(
                    self.trainer.model.test_history[0]['activities_mpjpe'], index=['mpjpe']
                ).T.to_dict(orient='records')
            }
            json.dumps(info, f)

    def test(self):
        self.trainer.test(
            ckpt_path=self.best_checkpoint_path,
            dataloaders=self.test_loader
        )
        self._write_test_results()
