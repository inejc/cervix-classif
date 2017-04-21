from math import ceil
from os.path import join

import fire
import numpy as np
from keras.applications.inception_v3 import \
    preprocess_input as inception_preprocess
from keras.applications.xception import preprocess_input as xception_preprocess
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from data_provider import load_organized_data_info, MODELS_DIR, SUBMISSIONS_DIR
from resnet50_fine_tune import preprocess_single_input as resnet50_preprocess
from vgg19_fine_tune import preprocess_single_input as vgg19_preprocess
from utils import create_submission_file
from xception_fine_tune import create_embeddings

WIDTH, HEIGHT = 299, 299
BATCH_SIZE = 32

MODELS = {
    'xception_fine_tuned_stable_frozen_86_dropout_0_2_val_loss_0_7288.h5':
        xception_preprocess,
    'xception_fine_tuned_stable_frozen_86_dropout_0_5_val_loss_0_7520.h5':
        xception_preprocess,
    'xception_fine_tuned_stable_frozen_86_dropout_0_6_val_loss_0_7386.h5':
        xception_preprocess,
    'xception_fine_tuned_stable_frozen_96_dropout_0_6_val_loss_0_7383.h5':
        xception_preprocess,
    'inception_fine_tuned_stable_frozen_280_dropout_0_5_val_loss_0_7203.h5':
        inception_preprocess,
    'inception_fine_tuned_stable_frozen_260_dropout_0_5_val_loss_0_7440.h5':
        inception_preprocess,
    'inception_fine_tuned_stable_frozen_250_dropout_0_5_val_loss_0_7473.h5':
        inception_preprocess,
    'resnet50_fine_tuned_stable_frozen_150_dropout_0_5_val_loss_0_7410.h5':
        resnet50_preprocess,
    'resnet50_fine_tuned_stable_frozen_140_dropout_0_5_val_loss_0_7365.h5':
        resnet50_preprocess,
    'resnet50_fine_tuned_stable_frozen_130_dropout_0_5_val_loss_0_6868.h5':
        resnet50_preprocess,
    'resnet50_fine_tuned_stable_frozen_120_dropout_0_5_val_loss_0_7174.h5':
        resnet50_preprocess,
    'vgg19_fine_tuned_stable_frozen_17_penultimate_256_dropout_0_5_val_loss_0_6631.h5':
        vgg19_preprocess,
    'vgg19_fine_tuned_stable_frozen_7_penultimate_256_dropout_0_5_val_loss_0_7142.h5':
        vgg19_preprocess,
}


def train(name='stable', cross_validate=True, num_search_iter=500):
    data_info = load_organized_data_info(imgs_dim=HEIGHT, name=name)

    preds_val = np.empty((data_info['num_val'], 0))
    if not cross_validate:
        preds_te = np.empty((data_info['num_te'], 0))

    for model_name, preprocess_func in MODELS.items():
        model_path = join(MODELS_DIR, model_name)

        model_preds_val = _make_predictions(
            model_path=model_path,
            preprocess_func=preprocess_func,
            data_info=data_info,
            dir_id='val'
        )

        if not cross_validate:
            model_preds_te = _make_predictions(
                model_path=model_path,
                preprocess_func=preprocess_func,
                data_info=data_info,
                dir_id='te'
            )

        preds_val = np.hstack((preds_val, model_preds_val))
        if not cross_validate:
            preds_te = np.hstack((preds_te, model_preds_te))

    _, _, _, y_val, _, te_names = create_embeddings(name=name)

    if cross_validate:
        l2_distribution = {'C': uniform(1e-3, 1e4)}
        random_search = RandomizedSearchCV(
            estimator=LogisticRegression(),
            param_distributions=l2_distribution,
            n_iter=num_search_iter,
            n_jobs=-1,
            cv=10,
        )
        random_search.fit(preds_val, y_val)

        print("Best random search score and params:")
        print(random_search.best_score_)
        print(random_search.best_params_)
    else:
        lr = LogisticRegression(C=1e10)
        lr.fit(preds_val, y_val)
        y_pred = lr.predict_proba(preds_te)

        create_submission_file(
            te_names,
            y_pred,
            join(SUBMISSIONS_DIR, 'stacked.csv')
        )

        print(lr.coef_)


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
