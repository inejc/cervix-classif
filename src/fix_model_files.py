from os.path import join

import h5py

from data_provider import MODELS_DIR

MODELS = [
    'xception_fine_tuned_final_frozen_86_dropout_0_2.h5',
    'inception_fine_tuned_final_frozen_280_dropout_0_5.h5',
    'resnet50_fine_tuned_final_frozen_120_dropout_0_5.h5',
    'vgg19_fine_tuned_final_frozen_17_penultimate_256_dropout_0_5.h5',
    'vgg16_fine_tuned_final_frozen_6_penultimate_256_dropout_0_5.h5',
    'xception_fine_tuned_final_frozen_96_dropout_0_6.h5',
    'inception_fine_tuned_final_frozen_270_dropout_0_5.h5',
    'resnet50_fine_tuned_final_frozen_130_dropout_0_5.h5',
    'vgg19_fine_tuned_final_frozen_12_penultimate_512_dropout_0_5.h5',
    'vgg16_fine_tuned_final_frozen_11_penultimate_512_dropout_0_5.h5',
    'xception_fine_tuned_final_frozen_86_dropout_0_6.h5',
    'inception_fine_tuned_final_frozen_260_dropout_0_5.h5',
    'resnet50_fine_tuned_final_frozen_140_dropout_0_5.h5',
    'vgg19_fine_tuned_final_frozen_7_penultimate_512_dropout_0_5.h5',
    'vgg16_fine_tuned_final_frozen_11_penultimate_256_dropout_0_5.h5',
    'xception_fine_tuned_final_frozen_86_dropout_0_5.h5',
    'inception_fine_tuned_final_frozen_250_dropout_0_5.h5',
    'resnet50_fine_tuned_final_frozen_150_dropout_0_5.h5',
    'vgg19_fine_tuned_final_frozen_7_penultimate_256_dropout_0_5.h5',
    'vgg16_fine_tuned_final_frozen_6_penultimate_512_dropout_0_5.h5',
]

for model in MODELS:
    f = h5py.File(join(MODELS_DIR, model), 'r+')
    del f['optimizer_weights']
    f.close()
