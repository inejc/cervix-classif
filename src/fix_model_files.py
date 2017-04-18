from os.path import join

import h5py

from data_provider import MODELS_DIR

MODELS = [
    'xception_fine_tuned_stable_frozen_86_dropout_0_2_val_loss_0_7288.h5',
    'xception_fine_tuned_stable_frozen_86_dropout_0_3_val_loss_0_7494.h5',
    'xception_fine_tuned_stable_frozen_86_dropout_0_4_val_loss_0_7155.h5',
    'xception_fine_tuned_stable_frozen_86_dropout_0_5_val_loss_0_7520.h5',
    'xception_fine_tuned_stable_frozen_86_dropout_0_6_val_loss_0_7386.h5'
]

for model in MODELS:
    f = h5py.File(join(MODELS_DIR, model), 'r+')
    del f['optimizer_weights']
    f.close()
