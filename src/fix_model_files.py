from os.path import join

import h5py

from data_provider import MODELS_DIR

MODELS = [
    'resnet50_fine_tuned_stable_frozen_120_dropout_0_5_val_loss_0_7174.h5',
    'resnet50_fine_tuned_stable_frozen_130_dropout_0_5_val_loss_0_6868.h5',
    'resnet50_fine_tuned_stable_frozen_140_dropout_0_5_val_loss_0_7365.h5',
    'resnet50_fine_tuned_stable_frozen_150_dropout_0_5_val_loss_0_7410.h5',
]

for model in MODELS:
    f = h5py.File(join(MODELS_DIR, model), 'r+')
    del f['optimizer_weights']
    f.close()
