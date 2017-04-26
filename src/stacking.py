from math import ceil
from os.path import join, isfile

import fire
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.externals.joblib import load, dump
from sklearn.linear_model import LogisticRegression

from data_provider import load_organized_data_info, MODELS_DIR
from xception_fine_tune import create_embeddings

BATCH_SIZE = 32


def stack(group):
    name = group['name'], width = group['width'], height = group['height']
    group_uid, models = group['uid'], group['models']

    meta_model_file = join(
        MODELS_DIR,
        'stacking_meta_model_group_{:d}.pickle'.format(group_uid)
    )
    meta_model_fitted = isfile(meta_model_file)

    data_info = load_organized_data_info(imgs_dim=width, name=name)

    if not meta_model_fitted:
        preds_val = np.empty((data_info['num_val'], 0))
    preds_te = np.empty((data_info['num_te'], 0))

    for model_name, preprocess_func in models:
        model_path = join(MODELS_DIR, model_name)

        if not meta_model_fitted:
            model_preds_val = _make_predictions(
                height=height,
                width=width,
                model_path=model_path,
                preprocess_func=preprocess_func,
                data_info=data_info,
                dir_id='val'
            )

        model_preds_te = _make_predictions(
            height=height,
            width=width,
            model_path=model_path,
            preprocess_func=preprocess_func,
            data_info=data_info,
            dir_id='te'
        )

        if not meta_model_fitted:
            preds_val = np.hstack((preds_val, model_preds_val))
        preds_te = np.hstack((preds_te, model_preds_te))

    # todo: use imgs dim param
    _, _, _, y_val, _, te_names = create_embeddings(name=name)

    if meta_model_fitted:
        meta_model = load(meta_model_file)
    else:
        meta_model = LogisticRegression(C=1e10)
        meta_model.fit(preds_val, y_val)
        dump(meta_model, meta_model_file)

    te_pred = meta_model.predict_proba(preds_te)
    return te_names, te_pred


def _make_predictions(height, width, model_path, preprocess_func,
                      data_info, dir_id):
    model = load_model(model_path)

    datagen = ImageDataGenerator(preprocessing_function=preprocess_func)
    datagen = datagen.flow_from_directory(
        directory=data_info['dir_' + dir_id],
        target_size=(height, width),
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
