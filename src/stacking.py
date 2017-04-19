from math import ceil
from os.path import join

import fire
import numpy as np
from keras.applications.xception import preprocess_input as xception_preprocess
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from data_provider import load_organized_data_info, MODELS_DIR, SUBMISSIONS_DIR
from utils import cross_val_scores, create_submission_file
from xception_fine_tune import create_embeddings

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
    'xception_fine_tuned_stable_frozen_96_dropout_0_3_val_loss_0_7326.h5':
        xception_preprocess,
    'xception_fine_tuned_stable_frozen_96_dropout_0_4_val_loss_0_7216.h5':
        xception_preprocess,
    'xception_fine_tuned_stable_frozen_96_dropout_0_5_val_loss_0_7669.h5':
        xception_preprocess,
    'xception_fine_tuned_stable_frozen_96_dropout_0_6_val_loss_0_7383.h5':
        xception_preprocess,
}


def train(name='stable', cross_validate=True, k=10):
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
        clfs = [
            ('stratified', DummyClassifier()),
            ('lr', LogisticRegression(C=1e10)),
            ('lr_l2', LogisticRegression(C=1)),
            ('svm', SVC(probability=True)),
            ('rf', RandomForestClassifier(n_estimators=500, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=500))
        ]

        scores = cross_val_scores(
            classifiers=clfs,
            X=preds_val,
            y=y_val,
            k=k
        )

        print(scores)
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
