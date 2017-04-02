from os import listdir
from os.path import join, splitext
from collections import OrderedDict

import fire
import ijroi
import numpy as np
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, load_img, \
    img_to_array

from data_dirs_organizer import HEIGHT, WIDTH
from data_provider import MODELS_DIR

IJ_ROI_DIR = join('..', 'data', 'bounding_boxes_299')
MODEL_FILE = join(MODELS_DIR, 'simple_localizer.h5')

CLASSES = ['Type_1', 'Type_2', 'Type_3']
TRAINING_DIR = join('..', 'data', 'train_299')


def _get_tagged_images():
    """Read image data.

    Return
    ------
    X : np.array
        Image data
    Y : np.array
        Bounding boxes in format [x, y, w, h]

    """
    roi_dict = {splitext(f)[0]: join(IJ_ROI_DIR, f)
                for f in listdir(IJ_ROI_DIR)}
    # Get the file names of the tagged image files
    img_dict = OrderedDict()
    for class_ in CLASSES:
        for f in listdir(join(TRAINING_DIR, class_)):
            img_id = splitext(f)[0]
            if img_id in roi_dict:
                img_dict[img_id] = join(TRAINING_DIR, class_, f)
    # Initialize X and Y (contains 4 values x, y, w, h)
    X = np.zeros((len(img_dict), HEIGHT, WIDTH, 3))
    Y = np.zeros((len(img_dict), 4))
    labels = []
    # Load the image files into a nice data array
    for idx, key in enumerate(img_dict):
        img = load_img(img_dict[key], target_size=(HEIGHT, WIDTH))
        labels.append(key)
        X[idx] = img_to_array(img)
        Y[idx] = _convert_from_roi(roi_dict[key])

    return labels, X, Y


def _convert_from_roi(fname):
    """Convert a roi file to a numpy array [x, y, w, h].

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

        return np.array([top, left, width, height])


def _cnn():
    model = Sequential()

    model.add(Conv2D(32, 3, 3, input_shape=(HEIGHT, WIDTH, 3),
                     activation='relu', border_mode='same'))
    model.add(Conv2D(32, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(Conv2D(64, 3, 3, activation='relu', border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4))

    return model


def train_simple(reduce_lr_factor=1e-1, epochs=10):
    _, X, Y = _get_tagged_images()

    def _image_generator():
        return generator.flow(
            X, Y,
            batch_size=32,
            shuffle=True,
        )

    model = _cnn()
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


if __name__ == '__main__':
    fire.Fire()
