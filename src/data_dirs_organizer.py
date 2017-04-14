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

from data_provider import DATA_DIR, CLASSES, IMAGES_WEIGHTS_FILE
from data_provider import num_examples_per_class_in_dir
from data_provider import organized_data_info_file
from data_provider import save_organized_data_info, load_organized_data_info
from utils import read_lines


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
    manner, train dir should be the first arg to tr_dirs (additional, gan, ...
    should follow).
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

    num_per_cls_tr = num_examples_per_class_in_dir(new_dir_tr)
    num_tr = sum(num_per_cls_tr.values())
    print("Organized training set class distribution:")
    print({k: v / num_tr for k, v in num_per_cls_tr.items()})

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

    train_paths, train_labels = _load_paths_labels_from_train_dir(dirs[0])

    other_paths, other_labels = [], []
    for dir_ in dirs[1:]:
        dir_paths, dir_labels = _load_paths_labels_from_train_dir(dir_)
        other_paths = np.hstack((other_paths, dir_paths))
        other_labels = np.hstack((other_labels, dir_labels))

    ind_tr, ind_val = _train_val_split_indices(
        val_size_fraction,
        train_paths,
        train_labels
    )

    # use only data from train dir for validation
    all_train_paths = np.hstack((train_paths[ind_tr], other_paths))
    all_train_labels = np.hstack((train_labels[ind_tr], other_labels))
    val_paths = train_paths[ind_val]
    val_labels = train_labels[ind_val]

    # duplicate weighted examples
    weighted_examples = read_lines(
        IMAGES_WEIGHTS_FILE,
        line_func=lambda l: l.rstrip()
    )

    weighted_paths, weighted_labels = [], []
    for w_example in weighted_examples:
        weighted_paths.append(join(DATA_DIR, w_example))
        weighted_labels.append(basename(dirname(w_example)))

    _save_images_to_dir(
        imgs_dim,
        new_dir_tr,
        np.array(weighted_paths),
        np.array(weighted_labels),
        names_ext='weighted'
    )

    _save_images_to_dir(imgs_dim, new_dir_tr, all_train_paths, all_train_labels)
    _save_images_to_dir(imgs_dim, new_dir_val, val_paths, val_labels)

    _save_organized_data_info(
        imgs_dim,
        name,
        len(all_train_paths) + len(weighted_examples),
        len(val_paths),
        new_dir_tr,
        new_dir_val
    )


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


def _save_images_to_dir(imgs_dim, dest_dir, src_paths, labels, names_ext=None):
    for src_path, label in zip(src_paths, labels):
        if names_ext is not None:
            file_name = '{:s}_{:s}_{:s}'.format(
                basename(dirname(dirname(src_path))),
                names_ext,
                basename(src_path)
            )
        else:
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
    try:
        image = load_img(src)
    except FileNotFoundError:
        print("Image {:s} not found and skipped".format(src))
        return

    # todo: preprocess
    image = fit(image, (imgs_dim, imgs_dim), method=LANCZOS)
    image.save(dest)


if __name__ == '__main__':
    fire.Fire()
