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
from keras.preprocessing.image import ImageDataGenerator

from data_provider import DATA_DIR, num_examples_per_class_in_dir
from data_provider import MODELS_DIR, load_organized_data_info

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
    return X_tr, y_tr, X_val, y_val, X_te, te_names


def train_top_classifier():
    # todo
    pass


def fine_tune():
    # todo
    pass


if __name__ == '__main__':
    fire.Fire()
