import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from src.modules.lifter_2d_3d.model.linear_model.linear_model import BaselineModel
from src.modules.lifter_2d_3d.dataset.groundtruth_keypoint_dataset import GroundTruthKeypointDataset
from pytorch_lightning.callbacks import TQDMProgressBar
import numpy as np


class LitSimpleBaselineLinear(pl.LightningModule):
    def __init__(self, exclude_ankle=False, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = BaselineModel(exclude_ankle=exclude_ankle)
        self.learning_rate = learning_rate
        self.val_losses = []
        self.val_print_count = 0
        self.train_loss_log = []

    def forward(self, x, batch_idx):
        # use forward for inference/predictions        
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
        # print(f"train batch_idx={batch_idx}, x.shape={x.shape}, y.shape={y.shape}")
        x = x.float().squeeze(2).to(self.device)
        y = y.float().squeeze(2).to(self.device)
        y_hat = self.model(x)
        # print(f"batch_idx={batch_idx}, y_hat.shape={y_hat.shape}")
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        self.train_loss_log.append(torch.sqrt(loss).item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # print(f"val batch_idx={batch_idx}, x.shape={x.shape}, y.shape={y.shape}")
        x = x.float().squeeze(2).to(self.device)
        y = y.float().squeeze(2).to(self.device)
        y_hat = self.model(x)
        # print(f"batch_idx={batch_idx}, y_hat.shape={y_hat.shape}")
        loss = F.mse_loss(y_hat, y)
        self.log("valid_loss", torch.sqrt(loss).item(), on_step=True)
        self.val_losses.append(torch.sqrt(loss).item())
        # raise Exception('hello')
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        # print(f"test batch_idx={batch_idx}, x.shape={x.shape}, y.shape={y.shape}")
        x = x.float().squeeze(2).to(self.device)
        y = y.float().squeeze(2).to(self.device)
        y_hat = self.model(x)
        # print(f"batch_idx={batch_idx}, y_hat.shape={y_hat.shape}")
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", torch.sqrt(loss).item())
        return loss

    # def on_validation_epoch_end(self):
    #     print(f"{self.val_print_count}: {len(self.val_losses)} batches : average val loss {sum(self.val_losses) / len(self.val_losses) * 1000}")
    #     self.val_print_count += 1
    #     self.val_losses = []

    def on_validation_epoch_end(self):
        print(f'check #{self.val_print_count}')
        print(f'training loss from {len(self.train_loss_log)} batches: {np.mean(self.train_loss_log) * 1000}')
        print(f"val loss from: {len(self.val_losses)} batches : {np.mean(self.val_losses) * 1000}")
        self.train_loss_log = []
        self.val_print_count += 1
        self.val_losses = []


    # def configure_optimizers(self):
    #     # self.hparams available because we called self.save_hyperparameters()
    #     # return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
    #     return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size = 25, gamma = 0.9
        )
        #learning rate scheduler
        return {
            "optimizer":optimizer,
            "lr_scheduler" : {
                "scheduler" : sch,
                "monitor" : "train_loss",
            }
        }

    # @staticmethod
    # def add_model_specific_args(parent_parser):
    #     parser = ArgumentParser(parents=[parent_parser], add_help=False)
    #     parser.add_argument('--learning_rate', type=float, default=0.0001)
    #     return parser


# def cli_main():
#     pl.seed_everything(1234)

#     # ------------
#     # args
#     # ------------
#     # parser = ArgumentParser()
#     # parser.add_argument('--batch_size', default=32, type=int)
#     # parser.add_argument('--hidden_dim', type=int, default=128)
#     # parser = pl.Trainer.add_argparse_args(parser)
#     # parser = LitClassifier.add_model_specific_args(parser)
#     # args = parser.parse_args()

#     # ------------
#     # data
#     # ------------
#     # dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
#     # mnist_test = MNIST('', train=False, download=True, transform=transforms.ToTensor())
#     # mnist_train, mnist_val = random_split(dataset, [55000, 5000])

#     train_dataset = GroundTruthKeypointDataset(
#         "/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/annotation/person_keypoints_train.json",
#         image_width=1280,
#         image_height=1024
#     )
#     val_dataset = GroundTruthKeypointDataset(
#         "/root/data/processed/synthetic_cabin_bw/A_Pillar_Codriver/annotation/person_keypoints_val.json",
#         image_width=1280,
#         image_height=1024
#     )
#     print('train_dataset', len(train_dataset), 'val_dataset', len(val_dataset))
#     train_loader = DataLoader(train_dataset, batch_size=2, drop_last=True)
#     val_loader = DataLoader(val_dataset, batch_size=2, drop_last=True)
#     # test_loader = DataLoader(mnist_test, batch_size=args.batch_size)

#     # ------------
#     # model
#     # ------------
#     # model = LitClassifier(Backbone(hidden_dim=args.hidden_dim), args.learning_rate)
#     lit_model = LitSimpleBaselineLinear()
#     # ------------
#     # training
#     # ------------
#     trainer = pl.Trainer(
#         # max_steps=10,
#         max_epochs=10,
#         # callbacks=[TQDMProgressBar(refresh_rate=5)],
#         val_check_interval=0.5,
#         # accelerator='gpu' if torch.cuda.is_available() else 'cpu',
#         accelerator='cpu',
#         # check_val_every_n_epoch=1,
#         # default_root_dir=saved_model_path
#     )
#     trainer.fit(lit_model, train_loader, val_loader)
#     # ------------
#     # testing
#     # ------------
#     # result = trainer.test(test_dataloaders=test_loader)
#     # print(result)


# if __name__ == "__main__":
#     cli_main()

