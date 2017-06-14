from os.path import join

import h5py

from data_provider import MODELS_DIR

MODELS = [
    'vgg16_fine_tuned_stable_frozen_11_penultimate_256_dropout_0_5_val_loss_0_7105.h5',
    'vgg16_fine_tuned_stable_frozen_6_penultimate_256_dropout_0_5_val_loss_0_6863.h5',
    'vgg16_fine_tuned_stable_frozen_3_penultimate_256_dropout_0_5_val_loss_0_7231.h5',
    'vgg16_fine_tuned_stable_frozen_11_penultimate_512_dropout_0_5_val_loss_0_7298.h5',
    'vgg16_fine_tuned_stable_frozen_6_penultimate_512_dropout_0_5_val_loss_0_7330.h5',
    'vgg16_fine_tuned_stable_frozen_3_penultimate_512_dropout_0_5_val_loss_0_7377.h5',
]

for model in MODELS:
    f = h5py.File(join(MODELS_DIR, model), 'r+')
    del f['optimizer_weights']
    f.close()
