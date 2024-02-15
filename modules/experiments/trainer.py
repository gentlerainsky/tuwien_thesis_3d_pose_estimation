import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# common setting
max_epoch = 200
val_check_period = 3
early_stopping_patience = 3

def create_trainer(
    saved_model_path,
    monitor_metric='mpjpe',
    max_epoch=max_epoch,
    val_check_period=val_check_period,
    early_stopping_patience=early_stopping_patience,
    enable_progress_bar=False,
    num_sanity_val_steps=-1,
):
    model_checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric, mode='min', save_top_k=1
    )
    early_stopping = EarlyStopping(
        monitor=monitor_metric,  mode="min", patience=early_stopping_patience
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not os.path.exists(saved_model_path):
        os.makedirs(saved_model_path)
    trainer = pl.Trainer(
        # max_steps=10,
        max_epochs=max_epoch,
        callbacks=[model_checkpoint_callback, early_stopping],
        accelerator=device,
        check_val_every_n_epoch=val_check_period,
        default_root_dir=saved_model_path,
        gradient_clip_val=1.0,
        logger=enable_progress_bar,
        enable_progress_bar=enable_progress_bar,
        num_sanity_val_steps=num_sanity_val_steps,
        log_every_n_steps=1
    )
    return dict(
        trainer=trainer,
        model_checkpoint_callback=model_checkpoint_callback
    )
