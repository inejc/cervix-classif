from os.path import join

import h5py

from data_provider import MODELS_DIR

MODELS = [
    'inception_fine_tuned_stable_frozen_250_dropout_0_5_val_loss_0_7473.h5',
    'inception_fine_tuned_stable_frozen_260_dropout_0_5_val_loss_0_7440.h5',
    'inception_fine_tuned_stable_frozen_270_dropout_0_5_val_loss_0_7166.h5',
    'inception_fine_tuned_stable_frozen_280_dropout_0_5_val_loss_0_7203.h5',
]

for model in MODELS:
    f = h5py.File(join(MODELS_DIR, model), 'r+')
    del f['optimizer_weights']
    f.close()
