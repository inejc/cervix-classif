"""
Usage:
    python xception_fine_tune.py create_embeddings
    python xception_fine_tune.py train_top_classifier
    python xception_fine_tune.py fine_tune
"""

from math import ceil
from os import listdir
from os.path import join, isfile

import fire
import numpy as np
from keras.applications.xception import Xception, preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical

from data_provider import DATA_DIR, num_examples_per_class_in_dir
from data_provider import EXPERIMENTS_DIR, SUBMISSIONS_DIR
from data_provider import MODELS_DIR, load_organized_data_info
from utils import create_submission_file

HEIGHT, WIDTH = 299, 299

MODEL_FILE = join(MODELS_DIR, 'xception_fine_tuned.h5')
TOP_CLASSIFIER_FILE = join(MODELS_DIR, 'xception_top_classifier.h5')
EMBEDDINGS_FILE = join(DATA_DIR, 'xception_embeddings.npz')


def create_embeddings():
    """Returns xception embeddings (outputs of the last conv layer).

    Returns
    -------
    tuple
        X_tr (n_samples, 2048)
        y_tr (n_samples,)
        X_val (n_samples, 2048)
        y_val (n_samples,)
        X_te (n_samples, 2048)
        te_names (n_samples,)
    """
    if isfile(EMBEDDINGS_FILE):
        d = np.load(EMBEDDINGS_FILE)
        return d['X_tr'], d['y_tr'], d['X_val'], d['y_val'], d['X_te'],\
            d['te_names']

    data_info = load_organized_data_info(imgs_dim=HEIGHT)
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    batch_size = 32

    def dir_datagen(dir_):
        return datagen.flow_from_directory(
            directory=dir_,
            target_size=(HEIGHT, WIDTH),
            class_mode=None,
            batch_size=batch_size,
            shuffle=False
        )

    model = Xception(weights='imagenet', include_top=False, pooling='avg')

    def embed(dir_, num, data_is_labeled):
        X = model.predict_generator(
            generator=dir_datagen(dir_),
            steps=ceil(num / batch_size)
        )

        if data_is_labeled:
            num_per_cls = num_examples_per_class_in_dir(dir_)
            y_0 = np.zeros(num_per_cls['Type_1'])
            y_1 = np.zeros(num_per_cls['Type_2']) + 1
            y_2 = np.zeros(num_per_cls['Type_3']) + 2
            y = np.hstack((y_0, y_1, y_2))
            return X, y

        # unlabeled (test) dataset
        names = [x for x in listdir(join(dir_, 'all')) if x.endswith('.jpg')]
        return X, np.array(names)

    dir_tr, num_tr = data_info['dir_tr'], data_info['num_tr']
    X_tr, y_tr = embed(dir_tr, num_tr, data_is_labeled=True)

    dir_val, num_val = data_info['dir_val'], data_info['num_val']
    X_val, y_val = embed(dir_val, num_val, data_is_labeled=True)

    dir_te, num_te = data_info['dir_te'], data_info['num_te']
    X_te, te_names = embed(dir_te, num_te, data_is_labeled=False)

    np.savez_compressed(
        file=EMBEDDINGS_FILE,
        X_tr=X_tr,
        y_tr=y_tr,
        X_val=X_val,
        y_val=y_val,
        X_te=X_te,
        te_names=te_names
    )

    print("Embedded data shapes:")
    print("X_tr {0}".format(X_tr.shape))
    print("y_tr {0}".format(y_tr.shape))
    print("X_val {0}".format(X_val.shape))
    print("y_val {0}".format(y_val.shape))
    print("X_te {0}".format(X_te.shape))
    print("te_names {0}".format(te_names.shape))
    return X_tr, y_tr, X_val, y_val, X_te, te_names


def train_top_classifier(lr=0.01, epochs=10, batch_size=32,
                         l2_reg=0, save_model=True):

    X_tr, y_tr, X_val, y_val, _, _ = create_embeddings()
    y_tr, y_val = to_categorical(y_tr), to_categorical(y_val)

    model = _top_classifier(l2_reg, X_tr.shape[1:])
    model.compile(Adam(lr=lr), loss='categorical_crossentropy')

    model.fit(
        X_tr, y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )

    if save_model:
        model.save(TOP_CLASSIFIER_FILE)


def make_submission_top_classifier():
    _, _, _, _, X_te, te_names = create_embeddings()

    model = _top_classifier(l2_reg=0, input_shape=X_te.shape[1:])
    model.load_weights(TOP_CLASSIFIER_FILE)

    probs_pred = model.predict_proba(X_te)
    create_submission_file(
        image_names=te_names,
        probs=probs_pred,
        file_name=join(SUBMISSIONS_DIR, 'xception_top_classifier.csv')
    )


def fine_tune(lr=1e-4, reduce_lr_factor=0.1, epochs=10, batch_size=32, l2_reg=0,
              num_freeze_layers=0):

    data_info = load_organized_data_info(HEIGHT)
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    batch_size = 32

    def dir_datagen(dir_):
        return datagen.flow_from_directory(
            directory=dir_,
            target_size=(HEIGHT, WIDTH),
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True
        )

    dir_tr, num_tr = data_info['dir_tr'], data_info['num_tr']
    dir_val, num_val = data_info['dir_val'], data_info['num_val']

    model = Xception(weights='imagenet', include_top=False, pooling='avg')
    top_classifier = _top_classifier(l2_reg, input_shape=(2048,))
    top_classifier.load_weights(TOP_CLASSIFIER_FILE)
    model = Model(inputs=model.input, outputs=top_classifier(model.output))
    model.compile(Adam(lr=lr), loss='categorical_crossentropy')

    # model has 134 layers
    for layer in model.layers[:num_freeze_layers]:
        layer.trainable = False

    callbacks = [
        ReduceLROnPlateau(factor=reduce_lr_factor),
        ModelCheckpoint(MODEL_FILE, save_best_only=True),
        TensorBoard(
            log_dir=join(EXPERIMENTS_DIR, 'xception_fine_tune'),
            write_graph=False
        )
    ]

    model.fit_generator(
        generator=dir_datagen(dir_tr),
        steps_per_epoch=ceil(num_tr / batch_size),
        epochs=epochs,
        validation_data=dir_datagen(dir_val),
        validation_steps=ceil(num_val / batch_size),
        callbacks=callbacks
    )


def _top_classifier(l2_reg, input_shape):
    model = Sequential()
    dense = Dense(
        units=3,
        kernel_regularizer=l2(l=l2_reg),
        activation='softmax',
        input_shape=input_shape
    )
    model.add(dense)
    return model


if __name__ == '__main__':
    fire.Fire()
