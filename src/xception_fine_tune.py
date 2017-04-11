"""
Example usage
-------------
    python xception_fine_tune.py create_embeddings --name 'cleaned'

    python xception_fine_tune.py train_top_classifier --name 'cleaned'
                                 --lr 0.0001 --epochs 3 --batch_size 32
                                 --l2_reg 0 --dropout_p 0.5 --save_model=True

    python xception_fine_tune.py make_submission_top_classifier --name 'cleaned'
                                 --dropout_p 0.5

    python xception_fine_tune.py fine_tune --name 'cleaned' --lr 1e-4
                                 --reduce_lr_factor 0.1 --reduce_lr_patience 3
                                 --epochs 2 --batch_size 32 --l2_reg 0
                                 --dropout_p 0.5 --num_freeze_layers 133

    python xception_fine_tune.py make_submission_xception --name 'cleaned'
                                 --dropout_p 0.5
"""

from math import ceil
from os import listdir
from os.path import join, isfile

import fire
import numpy as np
from keras.applications.xception import Xception, preprocess_input
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from keras.layers import Dense, Dropout
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical

from data_provider import DATA_DIR, num_examples_per_class_in_dir
from data_provider import EXPERIMENTS_DIR, SUBMISSIONS_DIR
from data_provider import MODELS_DIR, load_organized_data_info
from utils import create_submission_file

HEIGHT, WIDTH = 299, 299
EMBEDDINGS_FILE = 'xception_embeddings_{:s}.npz'
TOP_CLASSIFIER_FILE = 'xception_top_classifier_{:s}.h5'
MODEL_FILE = 'xception_fine_tuned_{:s}_{:s}.h5'


def create_embeddings(name):
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
    embeddings_file = join(DATA_DIR, EMBEDDINGS_FILE.format(name))

    if isfile(embeddings_file):
        d = np.load(embeddings_file)
        return d['X_tr'], d['y_tr'], d['X_val'], d['y_val'], d['X_te'],\
            d['te_names']

    data_info = load_organized_data_info(imgs_dim=HEIGHT, name=name)
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
        file=embeddings_file,
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


def train_top_classifier(name, lr=0.01, epochs=10, batch_size=32,
                         l2_reg=0, dropout_p=0.5, save_model=True):

    X_tr, y_tr, X_val, y_val, _, _ = create_embeddings(name)
    y_tr, y_val = to_categorical(y_tr), to_categorical(y_val)

    model_file = join(MODELS_DIR, TOP_CLASSIFIER_FILE.format(name))
    model = _top_classifier(
        l2_reg=l2_reg,
        dropout_p=dropout_p,
        input_shape=X_tr.shape[1:]
    )
    model.compile(Adam(lr=lr), loss='categorical_crossentropy')

    model.fit(
        X_tr, y_tr,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val)
    )

    if save_model:
        model.save(model_file)


def make_submission_top_classifier(name, dropout_p):
    _, _, _, _, X_te, te_names = create_embeddings(name)

    model_file = join(MODELS_DIR, TOP_CLASSIFIER_FILE.format(name))
    model = _top_classifier(
        l2_reg=0,
        dropout_p=dropout_p,
        input_shape=X_te.shape[1:]
    )
    model.load_weights(model_file)

    probs_pred = model.predict_proba(X_te)

    submission_file = 'xception_top_classifier_{:s}.csv'.format(name)
    create_submission_file(
        image_names=te_names,
        probs=probs_pred,
        file_name=join(SUBMISSIONS_DIR, submission_file)
    )


def fine_tune(name, name_ext, lr=1e-4, reduce_lr_factor=0.1,
              reduce_lr_patience=3, epochs=10, batch_size=32, l2_reg=0,
              dropout_p=0.5, num_freeze_layers=0, save_best_only=True):

    data_info = load_organized_data_info(imgs_dim=HEIGHT, name=name)
    tr_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=180,
        vertical_flip=True,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=0.3,
        # fill_mode='reflect'
    )
    val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    batch_size = 32

    def dir_datagen(dir_, gen):
        return gen.flow_from_directory(
            directory=dir_,
            target_size=(HEIGHT, WIDTH),
            class_mode='categorical',
            batch_size=batch_size,
            shuffle=True
        )

    dir_tr, num_tr = data_info['dir_tr'], data_info['num_tr']
    dir_val, num_val = data_info['dir_val'], data_info['num_val']

    top_classifier_file = join(MODELS_DIR, TOP_CLASSIFIER_FILE.format(name))
    model_file = join(MODELS_DIR, MODEL_FILE.format(name, name_ext))

    model = Xception(weights='imagenet', include_top=False, pooling='avg')
    top_classifier = _top_classifier(
        l2_reg=l2_reg,
        dropout_p=dropout_p,
        input_shape=(2048,)
    )
    top_classifier.load_weights(top_classifier_file)
    model = Model(inputs=model.input, outputs=top_classifier(model.output))
    model.compile(Adam(lr=lr), loss='categorical_crossentropy')

    # model has 134 layers
    for layer in model.layers[:num_freeze_layers]:
        layer.trainable = False

    log_dir = join(EXPERIMENTS_DIR, 'xception_fine_tuned_{:s}'.format(name))
    callbacks = [
        ReduceLROnPlateau(factor=reduce_lr_factor, patience=reduce_lr_patience),
        ModelCheckpoint(model_file, save_best_only=save_best_only),
        TensorBoard(
            log_dir=log_dir,
            write_graph=False
        )
    ]

    model.fit_generator(
        generator=dir_datagen(dir_tr, tr_datagen),
        steps_per_epoch=ceil(num_tr / batch_size),
        epochs=epochs,
        validation_data=dir_datagen(dir_val, val_datagen),
        validation_steps=ceil(num_val / batch_size),
        callbacks=callbacks
    )


def make_submission_xception(name, name_ext, dropout_p):
    data_info = load_organized_data_info(imgs_dim=HEIGHT, name=name)
    _, _, _, _, _, te_names = create_embeddings(name)
    batch_size = 32

    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    datagen = datagen.flow_from_directory(
        directory=data_info['dir_te'],
        target_size=(HEIGHT, WIDTH),
        class_mode=None,
        batch_size=batch_size,
        shuffle=False
    )

    model_file = join(MODELS_DIR, MODEL_FILE.format(name, name_ext))
    model = Xception(weights='imagenet', include_top=False, pooling='avg')
    top_classifier = _top_classifier(
        l2_reg=0,
        dropout_p=dropout_p,
        input_shape=(2048,)
    )
    model = Model(inputs=model.input, outputs=top_classifier(model.output))
    model.load_weights(model_file)

    probs_pred = model.predict_generator(
        generator=datagen,
        steps=ceil(data_info['num_te'] / batch_size)
    )

    submission_file = 'xception_fine_tuned_{:s}.csv'.format(name)
    create_submission_file(
        image_names=te_names,
        probs=probs_pred,
        file_name=join(SUBMISSIONS_DIR, submission_file)
    )


def _top_classifier(l2_reg, dropout_p, input_shape):
    model = Sequential()
    model.add(Dropout(rate=dropout_p, input_shape=input_shape))
    dense = Dense(
        units=3,
        kernel_regularizer=l2(l=l2_reg),
        activation='softmax'
    )
    model.add(dense)
    return model


if __name__ == '__main__':
    fire.Fire()
