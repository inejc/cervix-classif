from os.path import join

import h5py

from data_provider import MODELS_DIR

MODELS = [
    'xception_fine_tuned_stable_frozen_106_dropout_0_5_val_loss_0_7550.h5',
    'xception_fine_tuned_stable_frozen_96_dropout_0_5_val_loss_0_7359.h5',
    'xception_fine_tuned_stable_frozen_76_dropout_0_5_val_loss_0_7702.h5',
    'xception_fine_tuned_stable_frozen_66_dropout_0_5_val_loss_0_7719.h5',
]

for model in MODELS:
    f = h5py.File(join(MODELS_DIR, model), 'r+')
    del f['optimizer_weights']
    f.close()
