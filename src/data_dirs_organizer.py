"""
Example usage
-------------
    python data_dirs_organizer.py organize --imgs_dim 299 --name 'cleaned'
                                  --val_size_fraction 0.1 --te_dir 'test'
                                  'train' 'additional'

    python data_dirs_organizer.py clean --imgs_dim 299 --name 'cleaned'
"""

from os import mkdir, listdir, makedirs, remove
from os.path import join, abspath, basename, dirname
from shutil import rmtree

import fire
import numpy as np
from PIL.Image import LANCZOS
from PIL.ImageOps import fit
from keras.preprocessing.image import load_img
from sklearn.model_selection import StratifiedShuffleSplit

from data_provider import DATA_DIR, CLASSES
from data_provider import organized_data_info_file
from data_provider import save_organized_data_info, load_organized_data_info


def clean(imgs_dim=299, name=''):
    """Deletes all resized images datasests (i.e. train, val, test) and the
    info file.
    """
    data_info = load_organized_data_info(imgs_dim, name)
    rmtree(data_info['dir_tr'])
    rmtree(data_info['dir_val'])
    rmtree(data_info['dir_te'])
    remove(organized_data_info_file(imgs_dim, name))


def organize(imgs_dim=299, name='', val_size_fraction=0.1,
             te_dir=None, *tr_dirs):
    """Splits labeled images into training and validation sets in a stratified
    manner.
    """
    new_dir_tr = join(DATA_DIR, 'train_{:d}{:s}'.format(imgs_dim, '_' + name))
    new_dir_val = join(DATA_DIR, 'val_{:d}{:s}'.format(imgs_dim, '_' + name))

    _make_labeled_dir_structure(new_dir_tr)
    _make_labeled_dir_structure(new_dir_val)

    _organize_train_dirs(
        tr_dirs,
        val_size_fraction,
        imgs_dim,
        name,
        new_dir_tr,
        new_dir_val
    )

    if te_dir is not None:
        new_dir_te = join(
            DATA_DIR,
            'test_{:d}{:s}'.format(imgs_dim, '_' + name)
        )
        # put test images in 'all' dir for keras generator
        new_dir_te = join(new_dir_te, 'all')
        _organize_test_dir(imgs_dim, name, te_dir, new_dir_te)


def _make_labeled_dir_structure(dir_):
    mkdir(dir_)
    for class_ in CLASSES:
        class_dir = join(dir_, str(class_))
        mkdir(class_dir)


def _organize_train_dirs(dirs, val_size_fraction, imgs_dim, name, new_dir_tr,
                         new_dir_val):
    paths, labels = [], []
    for dir_ in dirs:
        dir_paths, dir_labels = _load_paths_labels_from_train_dir(dir_)
        paths = np.hstack((paths, dir_paths))
        labels = np.hstack((labels, dir_labels))

    ind_tr, ind_val = _train_val_split_indices(val_size_fraction, paths, labels)
    _save_organized_data_info(
        imgs_dim,
        name,
        len(ind_tr),
        len(ind_val),
        new_dir_tr,
        new_dir_val
    )

    _save_images_to_dir(imgs_dim, new_dir_tr, paths[ind_tr], labels[ind_tr])
    _save_images_to_dir(imgs_dim, new_dir_val, paths[ind_val], labels[ind_val])


def _load_paths_labels_from_train_dir(dir_):
    paths, labels = [], []

    for class_dir in CLASSES:
        for file_name in listdir(join(DATA_DIR, dir_, class_dir)):

            if not file_name.endswith('.jpg'):
                continue

            abspath_ = abspath(join(DATA_DIR, dir_, class_dir, file_name))
            paths.append(abspath_)
            labels.append(class_dir)

    return np.array(paths), np.array(labels)


def _train_val_split_indices(val_size_fraction, paths, labels):
    split = StratifiedShuffleSplit(
        n_splits=1, test_size=val_size_fraction, random_state=0)
    return next(split.split(paths, labels))


def _save_organized_data_info(imgs_dim, name, num_tr, num_val, new_dir_tr,
                              new_dir_val):
    info = {
        'dir_tr': new_dir_tr,
        'num_tr': num_tr,
        'dir_val': new_dir_val,
        'num_val': num_val,
        'num_classes': len(CLASSES),
    }
    save_organized_data_info(info, imgs_dim, name)


def _save_images_to_dir(imgs_dim, dest_dir, src_paths, labels):
    for src_path, label in zip(src_paths, labels):
        file_name = '{:s}_{:s}'.format(
            basename(dirname(dirname(src_path))),
            basename(src_path)
        )
        dest_path = join(join(dest_dir, label), file_name)
        _save_preprocessed_img(imgs_dim, src_path, dest_path)


def _organize_test_dir(imgs_dim, name, te_dir, new_dir_te):
    makedirs(new_dir_te)

    num_test_samples = 0
    for file_name in listdir(join(DATA_DIR, te_dir)):

        if not file_name.endswith('.jpg'):
            continue

        src_path = abspath(join(DATA_DIR, te_dir, file_name))
        dest_path = join(new_dir_te, file_name)
        _save_preprocessed_img(imgs_dim, src_path, dest_path)
        num_test_samples += 1

    _add_test_info_to_organized_data_info(
        imgs_dim,
        name,
        num_test_samples,
        new_dir_te
    )


def _add_test_info_to_organized_data_info(imgs_dim, name, num_test_samples,
                                          new_dir_te):
    data_info = load_organized_data_info(imgs_dim, name)
    data_info['dir_te'] = dirname(new_dir_te)
    data_info['num_te'] = num_test_samples
    save_organized_data_info(data_info, imgs_dim, name)


def _save_preprocessed_img(imgs_dim, src, dest):
    image = load_img(src)

    # todo: preprocess
    image = fit(image, (imgs_dim, imgs_dim), method=LANCZOS)
    image.save(dest)


if __name__ == '__main__':
    fire.Fire()
