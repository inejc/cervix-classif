from collections import OrderedDict
from os import listdir
from os.path import join, splitext

import fire
import ijroi
import numpy as np
from keras.applications import Xception
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense, Dropout
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator, load_img, \
    img_to_array
from keras.regularizers import l2

from xception_fine_tune import HEIGHT, WIDTH
from data_provider import MODELS_DIR
from xception_fine_tune import _top_classifier

IJ_ROI_DIR = join('..', 'data', 'bounding_boxes_299')
MODEL_FILE = join(MODELS_DIR, 'localizer.h5')

CLASSES = ['Type_1', 'Type_2', 'Type_3']
TRAINING_DIR = join('..', 'data', 'train_299')

__all__ = ['number_tagged', 'train', 'predict']


def _get_dict_roi():
    d = OrderedDict()
    for f in listdir(IJ_ROI_DIR):
        d[splitext(f)[0]] = join(IJ_ROI_DIR, f)
    return d


def _get_dict_all_images():
    d = OrderedDict()
    for class_ in CLASSES:
        for f in listdir(join(TRAINING_DIR, class_)):
            img_id = splitext(f)[0]
            d[img_id] = join(TRAINING_DIR, class_, f)
    return d


def _get_dict_tagged_images():
    all_images, tagged_roi = _get_dict_all_images(), _get_dict_roi()
    d = OrderedDict()
    for img_id in all_images:
        if img_id in tagged_roi:
            d[img_id] = all_images[img_id]
    return d


def _get_dict_untagged_images():
    d = _get_dict_all_images()
    for img_id in _get_dict_tagged_images():
        del d[img_id]
    return d


def _convert_from_roi(fname):
    """Convert a roi file to a numpy array [x, y, h, w].

    Parameters
    ----------
    fname : string
        If ends with `.roi`, we assume a full path is given

    """
    if not fname.endswith('.roi'):
        fname = '%s.roi' % join(IJ_ROI_DIR, fname)

    with open(fname, 'rb') as f:
        roi = ijroi.read_roi(f)
        top, left = roi[0]
        bottom, right = roi[2]
        height, width = bottom - top, right - left

        return np.array([top, left, height, width])


def _get_tagged_images():
    """Read images, tags and labels for any images that have been tagged.

    Return
    ------
    labels : array
    X : np.array
        Images
    Y : np.array
        Bounding boxes in format [y, x, h, w]

    """
    roi_dict, img_dict = _get_dict_roi(), _get_dict_tagged_images()
    # Initialize X and Y (contains 4 values x, y, w, h)
    X = np.zeros((len(img_dict), HEIGHT, WIDTH, 3))
    Y = np.zeros((len(img_dict), 4))
    # Load the image files into a nice data array
    for idx, key in enumerate(img_dict):
        img = load_img(img_dict[key], target_size=(HEIGHT, WIDTH))
        X[idx] = img_to_array(img)
        Y[idx] = _convert_from_roi(roi_dict[key])

    return list(img_dict.keys()), X, Y


def _load_images(fnames):
    X = np.zeros((len(fnames), HEIGHT, WIDTH, 3))
    for idx, fname in enumerate(fnames):
        X[idx] = load_img(join(TRAINING_DIR, fname))
    return fnames, X


def _get_untagged_images():
    img_dict = _get_dict_untagged_images()
    X = np.zeros((len(img_dict), HEIGHT, WIDTH, 3))
    for idx, img_id in enumerate(img_dict):
        X[idx] = load_img(img_dict[img_id])
    return list(img_dict.keys()), X


def _get_all_images():
    img_dict = _get_dict_all_images()
    X = np.zeros((len(img_dict), HEIGHT, WIDTH, 3))
    for idx, img_id in enumerate(img_dict):
        X[idx] = load_img(img_dict[img_id])
    return list(img_dict.keys()), X


def number_tagged():
    print('Number of tagged images', _get_tagged_images()[1].shape[0])
    print('Number of untagged images', _get_untagged_images()[1].shape[0])


def _cnn(model_file):
    # Load the classification model to get the trianed weights
    model = Xception(weights='imagenet', include_top=False, pooling='avg')
    top_classifier = _top_classifier(
        l2_reg=0,
        dropout_p=0.5,
        input_shape=(2048,)
    )
    model_ = Model(inputs=model.input, outputs=top_classifier(model.output))
    model_.load_weights(model_file)
    # Time to chop off the classification head and attach the regression head
    regression_head = _regression_head(
        l2_reg=0.0,
        dropout_p=0.5,
        input_shape=model.output,
    )
    return Model(inputs=model.input, outputs=regression_head(model.output))


def _regression_head(l2_reg, dropout_p, input_shape):
    model = Sequential()
    model.add(Dropout(rate=dropout_p, input_shape=input_shape))
    dense = Dense(
        units=4,
        kernel_regularizer=l2(l=l2_reg),
    )
    model.add(dense)
    return model


def train(reduce_lr_factor=1e-1, epochs=5):
    _, X, Y = _get_tagged_images()

    def _image_generator():
        return generator.flow(
            X, Y,
            batch_size=32,
            shuffle=True,
        )

    model = _cnn()
    # TODO See if an L1 loss does any better
    model.compile(loss='mean_squared_error', optimizer='adam')

    generator = ImageDataGenerator()
    callbacks = [
        ReduceLROnPlateau(factor=reduce_lr_factor),
        ModelCheckpoint(MODEL_FILE, save_best_only=True),
    ]
    model.fit_generator(
        generator=_image_generator(),
        steps_per_epoch=len(X),
        epochs=epochs,
        callbacks=callbacks,
        # TODO Don't validate on train data.
        validation_data=_image_generator(),
        validation_steps=len(X),
    )


def predict():
    model = _cnn()
    model.load_weights(MODEL_FILE)
    labels, X = _get_all_images()

    print(labels[:20])
    predictions = model.predict(X)

    print(predictions[:5])
    print(predictions.shape)

    np.save('predictions.npy', predictions)


if __name__ == '__main__':
    fire.Fire()
