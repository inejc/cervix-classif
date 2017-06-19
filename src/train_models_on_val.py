from math import ceil
from os.path import join

import fire
import h5py
from keras.applications.inception_v3 import \
    preprocess_input as inception_preprocess
from keras.applications.xception import preprocess_input as xception_preprocess
from keras.models import load_model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from data_provider import MODELS_DIR, load_organized_data_info
from resnet50_fine_tune import preprocess_single_input as resnet50_preprocess
from vgg16_fine_tune import preprocess_single_input as vgg16_preprocess
from vgg19_fine_tune import preprocess_single_input as vgg19_preprocess

NUM_EPOCHS = 3

MODELS = [
    ('xception_fine_tuned_final_frozen_86_dropout_0_2_val_trained.h5', xception_preprocess),
    ('inception_fine_tuned_final_frozen_280_dropout_0_5_val_trained.h5', inception_preprocess),
    ('resnet50_fine_tuned_final_frozen_120_dropout_0_5_val_trained.h5', resnet50_preprocess),
    ('vgg19_fine_tuned_final_frozen_17_penultimate_256_dropout_0_5_val_trained.h5', vgg19_preprocess),
    ('vgg16_fine_tuned_final_frozen_6_penultimate_256_dropout_0_5_val_trained.h5', vgg16_preprocess),
    ('xception_fine_tuned_final_frozen_96_dropout_0_6_val_trained.h5', xception_preprocess),
    ('inception_fine_tuned_final_frozen_270_dropout_0_5_val_trained.h5', inception_preprocess),
    ('resnet50_fine_tuned_final_frozen_130_dropout_0_5_val_trained.h5', resnet50_preprocess),
    ('vgg19_fine_tuned_final_frozen_12_penultimate_512_dropout_0_5_val_trained.h5', vgg19_preprocess),
    ('vgg16_fine_tuned_final_frozen_11_penultimate_512_dropout_0_5_val_trained.h5', vgg16_preprocess),
    ('xception_fine_tuned_final_frozen_86_dropout_0_6_val_trained.h5', xception_preprocess),
    ('inception_fine_tuned_final_frozen_260_dropout_0_5_val_trained.h5', inception_preprocess),
    ('resnet50_fine_tuned_final_frozen_140_dropout_0_5_val_trained.h5', resnet50_preprocess),
    ('vgg19_fine_tuned_final_frozen_7_penultimate_512_dropout_0_5_val_trained.h5', vgg19_preprocess),
    ('vgg16_fine_tuned_final_frozen_11_penultimate_256_dropout_0_5_val_trained.h5', vgg16_preprocess),
    ('xception_fine_tuned_final_frozen_86_dropout_0_5_val_trained.h5', xception_preprocess),
    ('inception_fine_tuned_final_frozen_250_dropout_0_5_val_trained.h5', inception_preprocess),
    ('resnet50_fine_tuned_final_frozen_150_dropout_0_5_val_trained.h5', resnet50_preprocess),
    ('vgg19_fine_tuned_final_frozen_7_penultimate_256_dropout_0_5_val_trained.h5', vgg19_preprocess),
    ('vgg16_fine_tuned_final_frozen_6_penultimate_512_dropout_0_5_val_trained.h5', vgg16_preprocess),
]


def train():
    batch_size = 32
    data_info = load_organized_data_info(imgs_dim=299, name='final')

    for model, preprocess_input in MODELS:
        tr_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=180,
            vertical_flip=True,
            horizontal_flip=True,
        )
        tr_datagen = tr_datagen.flow_from_directory(
            directory=data_info['dir_val'],
            target_size=(299, 299),
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True
        )

        model_file = join(MODELS_DIR, model)
        model = load_model(model_file)
        model.compile(Adam(lr=1e-5), loss='categorical_crossentropy')

        model.fit_generator(
            generator=tr_datagen,
            steps_per_epoch=ceil(data_info['num_val'] / batch_size),
            epochs=NUM_EPOCHS
        )
        model.save(model_file)
        f = h5py.File(model_file, 'r+')
        del f['optimizer_weights']
        f.close()


if __name__ == '__main__':
    fire.Fire()
