"""
Example usage:
CUDA_VISIBLE_DEVICES=0 python3 src/bounding_box.py train
    --name cleaned
    --model_file models/xception_fine_tuned_cleaned_frozen_96_dropout_0_6_val_loss_0_7404.h5
    --reg l1
    --reg_strength 0
    --dropout 0.5
    --epochs=10
"""
import re
from collections import OrderedDict
from os import listdir
from os.path import join, splitext

import fire
import ijroi
import numpy as np
from data_provider import MODELS_DIR, load_organized_data_info
from keras.applications import Xception
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard, \
    EarlyStopping
from keras.layers import Dense, Dropout
from keras.models import Sequential, Model, load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, \
    img_to_array
from keras.regularizers import l2, l1
from xception_fine_tune import HEIGHT, WIDTH
from xception_fine_tune import _top_classifier
import roi

IJ_ROI_DIR = join('data', 'bounding_boxes_299')
MODEL_FILE = 'localizer_.h5'

CLASSES = ['Type_1', 'Type_2', 'Type_3']
TRAINING_DIR = join('data', 'train_299')

__all__ = ['number_tagged', 'train', 'predict']


def _get_dict_roi(directory=None):
    """Get all available images with ROI bounding box.
    
    Returns
    -------
    dict : {<image_id>: <ROI file path>}
    
    """
    d = OrderedDict()
    for f in listdir(directory or IJ_ROI_DIR):
        d[splitext(f)[0]] = join(directory or IJ_ROI_DIR, f)
    return d


def _get_dict_all_images(directory=None, truncate_to_id=False):
    """Get all available images witch have an ROI bounding box label.
    
    Returns
    -------
    dict : {<image_id>: <image file path>}
    
    """
    d = OrderedDict()
    id_pattern = re.compile(r'\d+')
    for class_ in CLASSES:
        for f in listdir(join(directory or TRAINING_DIR, class_)):
            img_id = splitext(f)[0]
            # Because the clened images sometimes contain other information
            # e.g. additional or train, we want to extract only the id, so we
            # match on that.
            if truncate_to_id:
                img_id = id_pattern.search(img_id).group(0)
            d[img_id] = join(directory or TRAINING_DIR, class_, f)
    return d


def _get_dict_tagged_images(
        directory=None, roi_directory=None, truncate_to_id=False):
    """Get all available images in the training directory.
    
    Returns
    -------
    dict : {<image_id>: <image file path>}
    
    """
    all_images = _get_dict_all_images(directory, truncate_to_id)
    tagged_roi = _get_dict_roi(roi_directory)
    d = OrderedDict()
    for img_id in all_images:
        if img_id in tagged_roi:
            d[img_id] = all_images[img_id]
    return d


def _get_dict_untagged_images(directory=None):
    d = _get_dict_all_images(directory)
    for img_id in _get_dict_tagged_images(directory):
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


def _get_tagged_images(training_dir, roi_dir=None, truncate_to_id=False):
    """Read images, tags and labels for any images that have been tagged.

    Return
    ------
    labels : array
    X : np.array
        Images
    Y : np.array
        Bounding boxes in format [y, x, h, w]

    """
    roi_dict = _get_dict_roi(roi_dir or IJ_ROI_DIR)
    img_dict = _get_dict_tagged_images(training_dir, roi_dir, truncate_to_id)
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


def _get_all_images(directory=None):
    img_dict = _get_dict_all_images(directory)
    X = np.zeros((len(img_dict), HEIGHT, WIDTH, 3))
    for idx, img_id in enumerate(img_dict):
        X[idx] = load_img(img_dict[img_id])
    return list(img_dict.keys()), X


def number_tagged():
    print('Number of tagged images', _get_tagged_images()[1].shape[0])
    print('Number of untagged images', _get_untagged_images()[1].shape[0])


def _model_file_name(name, reg, reg_strength, dropout):
    split = MODEL_FILE.split('.')
    split[0] += '%s_%s-%s_dropout-%s_'% (name, reg, reg_strength, dropout)
    split[0] += 'val_loss-{val_loss:.2f}'
    return join(MODELS_DIR, '.'.join(split))


def _cnn(model_file, reg='l2', reg_strength=0.0, dropout_p=0.5):
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
        reg=reg,
        reg_strength=reg_strength,
        dropout_p=dropout_p,
        input_shape=(2048,),
    )
    return Model(inputs=model.input, outputs=regression_head(model.output))


def _regression_head(input_shape, reg='l2', reg_strength=0.0, dropout_p=0.5):
    model = Sequential()
    model.add(Dropout(rate=dropout_p, input_shape=input_shape))
    # Figure out regularization
    if reg == 'l2':
        regularization = l2
    elif reg == 'l1':
        regularization = l1
    # Create dense layer
    dense = Dense(
        units=4,
        kernel_regularizer=regularization(l=reg_strength),
    )
    model.add(dense)
    return model


def train(model_file, reduce_lr_factor=1e-1, num_freeze_layers=0, epochs=10,
          name='', reg='l2', reg_strength=0.0, dropout=0.5,
          early_stopping=False):
    data_info = load_organized_data_info(imgs_dim=HEIGHT, name=name)
    _, X_tr, Y_tr = _get_tagged_images(
        data_info['dir_tr'], truncate_to_id=True)
    _, X_val, Y_val = _get_tagged_images(
        data_info['dir_val'], truncate_to_id=True)

    def _image_generator(generator, data, labels):
        return generator.flow(
            data, labels,
            batch_size=32,
            shuffle=True,
        )

    model = _cnn(
        model_file=model_file,
        reg=reg,
        reg_strength=reg_strength,
        dropout_p=dropout
    )
    model.compile(loss='mean_squared_error', optimizer='adam')

    # model has 134 layers
    for layer in model.layers[:num_freeze_layers]:
        layer.trainable = False

    callbacks = [
        ReduceLROnPlateau(factor=reduce_lr_factor),
        ModelCheckpoint(_model_file_name(
            name, reg, reg_strength, dropout
        ), save_best_only=True),
        TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True),
    ]
    if early_stopping:
        callbacks.append(EarlyStopping(
            monitor='val_loss', min_delta=1, patience=early_stopping
        ))

    generator = ImageDataGenerator()
    model.fit_generator(
        generator=_image_generator(generator, X_tr, Y_tr),
        steps_per_epoch=len(X_tr),
        epochs=epochs,
        callbacks=callbacks,
        validation_data=_image_generator(generator, X_val, Y_val),
        validation_steps=len(X_val),
    )


def fix_model(model_file):
    import h5py
    f = h5py.File(model_file, 'r+')
    del f['optimizer_weights']
    f.close()


def resume_training(model_file, name='', reduce_lr_factor=1e-1,
                    num_freeze_layers=0, epochs=10, reg='l2', reg_strength=0.0,
                    dropout=0.5):
    data_info = load_organized_data_info(imgs_dim=HEIGHT, name=name)
    _, X_tr, Y_tr = _get_tagged_images(
        data_info['dir_tr'], truncate_to_id=True)
    _, X_val, Y_val = _get_tagged_images(
        data_info['dir_val'], truncate_to_id=True)

    def _image_generator(generator, data, labels):
        return generator.flow(
            data, labels,
            batch_size=32,
            shuffle=True,
        )
    model = load_model(model_file)

    for layer in model.layers[:num_freeze_layers]:
        layer.trainable = False

    callbacks = [
        ReduceLROnPlateau(factor=reduce_lr_factor),
        ModelCheckpoint(_model_file_name(
            name, reg, reg_strength, dropout
        ), save_best_only=True),
        TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True),
    ]
    generator = ImageDataGenerator()
    model.fit_generator(
        generator=_image_generator(generator, X_tr, Y_tr),
        steps_per_epoch=len(X_tr),
        epochs=epochs,
        callbacks=callbacks,
        validation_data=_image_generator(generator, X_val, Y_val),
        validation_steps=len(X_val),
    )


def predict(model_file, image_folder, out_dir=None):
    labels, X = _get_all_images(image_folder)
    model = load_model(model_file)

    predictions = model.predict(X)

    mask = predictions < 0
    predictions[mask] = 0
    mask = predictions > WIDTH
    predictions[mask] = WIDTH

    if out_dir:
        np.savez(join(out_dir, 'predictions.npz'))
        roi.save_predictions(labels, predictions, output_dir=out_dir)


if __name__ == '__main__':
    fire.Fire()
