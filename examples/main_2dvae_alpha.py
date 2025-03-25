"""Alpha VAE performs a weighted sum between two loss functions, given by
`loss = alpha * loss_func1 + (1 - alpha) * loss_func2`"""


import os

import torch
import torch.nn as nn
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from torchinfo import summary

from lvae3d.models.ResNetVAE_3Deu import ResNet18_3DVAEeu
from lvae3d.LightningVAETrainers import VAETrainerAlpha
from lvae3d.util.DataLoaders import Dataset3D, DataModule
from lvae3d.util.LossFunctions import SpectralLoss2D
from lvae3d.util.MetadataDicts import MetadataAlpha


if __name__ == '__main__':

    # Hyperparameters
    PATCH_SIZE = 64
    N_EPOCHS = 10
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 512
    LATENT_DIM = 32
    HIDDEN_DIM = 8192
    ALPHA = 0.999
    N_CHANNELS = 1
    PARALLEL = False
    AMSGRAD = True
    vae = ResNet18_3DVAEeu
    TrainerModule = VAETrainerAlpha
    loss_func1 = nn.BCEWithLogitsLoss()
    loss_func2 = SpectralLoss2D()

    # Filepaths
    TRAIN_DIR = 'path/to/train/data/'
    VAL_DIR = 'path/to/val/data'
    OUT_DIR = f'./out/{TrainerModule.__name__}/{vae.__name__}/latent{LATENT_DIM}/'
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Load checkpoint parameters
    load_model = False
    checkpoint_path = f'{OUT_DIR}/checkpoint.ckpt'

    # Torch config
    seed_everything(42, workers=True)
    torch.set_float32_matmul_precision('medium')

    # Metadata
    metadata = MetadataAlpha()
    if load_model:
        metadata.load(f'{OUT_DIR}metadata.yaml')
        metadata.metadata_dict['initial_epoch'] = metadata.metadata_dict['n_epochs'] + 1
        metadata.metadata_dict['n_epochs'] += N_EPOCHS
    else:
        metadata.create(vae=vae,
                        TrainerModule=TrainerModule,
                        loss_func1=loss_func1,
                        loss_func2=loss_func2,
                        alpha=ALPHA,
                        parallel=PARALLEL,
                        patch_size=PATCH_SIZE,
                        n_channels = N_CHANNELS,
                        n_epochs=N_EPOCHS,
                        learning_rate=LEARNING_RATE,
                        weight_decay=WEIGHT_DECAY,
                        batch_size=BATCH_SIZE,
                        latent_dim=LATENT_DIM,
                        hidden_dim=HIDDEN_DIM,
                        amsgrad=AMSGRAD
                        )

    # Load microstructure data
    train_dataset = Dataset3D(root_dir=TRAIN_DIR)
    val_dataset = Dataset3D(root_dir=VAL_DIR)
    data_module = DataModule(BATCH_SIZE, train_dataset, val_dataset, num_workers=8)

    # Train model
    if load_model:
        model = VAETrainerAlpha.load_from_checkpoint(checkpoint_path,
                                                     vae=vae,
                                                     metadata=metadata,
                                                     loss_func1=loss_func1,
                                                     loss_func2=loss_func2
                                                     )
    else:
        model = VAETrainerAlpha(vae, metadata, loss_func1, loss_func2)
    summary(model, (1, N_CHANNELS, PATCH_SIZE, PATCH_SIZE, PATCH_SIZE))

    checkpoint_callback = ModelCheckpoint(dirpath=OUT_DIR)
    trainer = Trainer(min_epochs=N_EPOCHS,
                      max_epochs=N_EPOCHS,
                      default_root_dir=OUT_DIR,
                      callbacks=[checkpoint_callback],
                      fast_dev_run=False,
                      accelerator='cpu',
                      log_every_n_steps=1,
                      num_sanity_val_steps=0,
                      )
    trainer.fit(model, data_module)
    trainer.save_checkpoint(filepath=f"{OUT_DIR}checkpoint_weights_epoch{metadata.metadata_dict['n_epochs']}.ckpt",
                            weights_only=True)

    metadata.save(f'{OUT_DIR}metadata.yaml')
