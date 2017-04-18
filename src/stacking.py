from math import ceil
from os.path import join

import fire
import numpy as np
from keras.applications.xception import preprocess_input as xception_preprocess
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

from data_provider import load_organized_data_info, MODELS_DIR

WIDTH, HEIGHT = 299, 299
BATCH_SIZE = 32

MODELS = {
    'xception_fine_tuned_stable_frozen_86_dropout_0_2_val_loss_0_7288.h5':
        xception_preprocess,
    'xception_fine_tuned_stable_frozen_86_dropout_0_3_val_loss_0_7494.h5':
        xception_preprocess,
    'xception_fine_tuned_stable_frozen_86_dropout_0_4_val_loss_0_7155.h5':
        xception_preprocess,
    'xception_fine_tuned_stable_frozen_86_dropout_0_5_val_loss_0_7520.h5':
        xception_preprocess,
    'xception_fine_tuned_stable_frozen_86_dropout_0_6_val_loss_0_7386.h5':
        xception_preprocess,
}


def train(name='stable'):
    data_info = load_organized_data_info(imgs_dim=HEIGHT, name=name)

    preds_val = np.empty((data_info['num_val'], 0))
    preds_te = np.empty((data_info['num_te'], 0))

    for model_name, preprocess_func in MODELS.items():
        model_path = join(MODELS_DIR, model_name)

        model_preds_val = _make_predictions(
            model_path=model_path,
            preprocess_func=preprocess_func,
            data_info=data_info,
            dir_id='val'
        )

        model_preds_te = _make_predictions(
            model_path=model_path,
            preprocess_func=preprocess_func,
            data_info=data_info,
            dir_id='te'
        )

        preds_val = np.hstack((preds_val, model_preds_val))
        preds_te = np.hstack((preds_te, model_preds_te))
        print(preds_val.shape)
        print(preds_te.shape)


def _make_predictions(model_path, preprocess_func, data_info, dir_id):
    model = load_model(model_path)

    datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
    datagen = datagen.flow_from_directory(
        directory=data_info['dir_' + dir_id],
        target_size=(HEIGHT, WIDTH),
        class_mode=None,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return model.predict_generator(
        generator=datagen,
        steps=ceil(data_info['num_' + dir_id] / BATCH_SIZE)
    )


if __name__ == '__main__':
    fire.Fire()
