from os import mkdir, listdir, makedirs, remove
from os.path import join, abspath, basename, dirname
from shutil import rmtree

import fire
import numpy as np
from PIL import ImageFile
from PIL.Image import LANCZOS
from PIL.ImageOps import fit
from keras.preprocessing.image import load_img
from sklearn.model_selection import StratifiedShuffleSplit

from data_provider import TRAIN_DIR, TEST_DIR, DATA_DIR, CLASSES
from data_provider import organized_data_info_file
from data_provider import save_organized_data_info, load_organized_data_info

HEIGHT, WIDTH = 299, 299

NEW_TRAIN_DIR = join(DATA_DIR, 'train_{:d}'.format(HEIGHT))
NEW_VAL_DIR = join(DATA_DIR, 'val_{:d}'.format(HEIGHT))
# put test images in 'all' dir for keras generator
NEW_TEST_DIR = join(DATA_DIR, 'test_{:d}'.format(HEIGHT))
NEW_TEST_DIR = join(NEW_TEST_DIR, 'all')

VAL_SIZE_FRACTION = 0.1

ImageFile.LOAD_TRUNCATED_IMAGES = True


def clean():
    data_info = load_organized_data_info(HEIGHT)
    rmtree(data_info['dir_tr'])
    rmtree(data_info['dir_val'])
    rmtree(data_info['dir_te'])
    remove(organized_data_info_file(HEIGHT))


def organize():
    _organize_train_dir()
    _organize_test_dir()


def _organize_train_dir():
    paths, labels = _load_paths_labels_from_train_dir()
    ind_tr, ind_val = _train_val_split_indices(paths, labels)
    _save_images_to_dir(NEW_TRAIN_DIR, paths[ind_tr], labels[ind_tr])
    _save_images_to_dir(NEW_VAL_DIR, paths[ind_val], labels[ind_val])


def _load_paths_labels_from_train_dir():
    paths, labels = [], []

    for class_dir in CLASSES:
        for file_name in listdir(join(TRAIN_DIR, class_dir)):

            if not file_name.endswith('.jpg'):
                continue

            abspath_ = abspath(join(TRAIN_DIR, class_dir, file_name))
            paths.append(abspath_)
            labels.append(class_dir)

    return np.array(paths), np.array(labels)


def _train_val_split_indices(paths, labels):
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=VAL_SIZE_FRACTION, random_state=0)
    indices_tr, indices_val = next(split.split(paths, labels))

    _save_organized_data_info(len(indices_tr), len(indices_val))
    return indices_tr, indices_val


def _save_organized_data_info(num_tr, num_val):
    info = {
        'dir_tr': NEW_TRAIN_DIR,
        'num_tr': num_tr,
        'dir_val': NEW_VAL_DIR,
        'num_val': num_val,
        'num_classes': len(CLASSES),
        'dir_te': dirname(NEW_TEST_DIR)
    }
    save_organized_data_info(info, HEIGHT)


def _save_images_to_dir(dest_dir, src_paths, labels):
    mkdir(dest_dir)

    for class_ in CLASSES:
        class_dir = join(dest_dir, str(class_))
        mkdir(class_dir)

    for src_path, label in zip(src_paths, labels):
        dest_path = join(join(dest_dir, label), basename(src_path))
        _save_scaled_cropped_img(src_path, dest_path)


def _organize_test_dir():
    makedirs(NEW_TEST_DIR)

    num_test_samples = 0
    for file_name in listdir(TEST_DIR):

        if not file_name.endswith('.jpg'):
            continue

        src_path = abspath(join(TEST_DIR, file_name))
        dest_path = join(NEW_TEST_DIR, file_name)
        _save_scaled_cropped_img(src_path, dest_path)
        num_test_samples += 1

    _append_num_te_to_organized_data_info(num_test_samples)


def _append_num_te_to_organized_data_info(num_test_samples):
    data_info = load_organized_data_info(HEIGHT)
    data_info['num_te'] = num_test_samples
    save_organized_data_info(data_info, HEIGHT)


def _save_scaled_cropped_img(src, dest):
    image = load_img(src)
    image = fit(image, (HEIGHT, WIDTH), method=LANCZOS)
    image.save(dest)


if __name__ == '__main__':
    fire.Fire()
