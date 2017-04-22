from os.path import join

import h5py

from data_provider import MODELS_DIR

MODELS = [
    'vgg19_fine_tuned_stable_frozen_17_penultimate_512_dropout_0_5_val_loss_0_7087.h5',
    'vgg19_fine_tuned_stable_frozen_12_penultimate_512_dropout_0_5_val_loss_0_6995.h5',
    'vgg19_fine_tuned_stable_frozen_7_penultimate_512_dropout_0_5_val_loss_0_6881.h5',
]

for model in MODELS:
    f = h5py.File(join(MODELS_DIR, model), 'r+')
    del f['optimizer_weights']
    f.close()
