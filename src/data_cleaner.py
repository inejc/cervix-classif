from os import mkdir, listdir, stat
from os.path import join, basename, dirname
from shutil import copyfile

import fire
from PIL import Image
from keras.preprocessing.image import img_to_array

from data_provider import DATA_DIR, CLASSES, IMAGES_BLACKLIST_FILE
from utils import read_lines, is_green


def clean(dir_):
    dir_cleaned = join(DATA_DIR, dir_ + '_cleaned')
    _make_labeled_dir_structure(dir_cleaned)
    dir_junk = join(DATA_DIR, dir_ + '_junk')
    _make_labeled_dir_structure(dir_junk)

    black_list = read_lines(
        IMAGES_BLACKLIST_FILE,
        line_func=lambda l: l.rstrip()
    )

    for class_dir in CLASSES:

        class_dir_abs = join(DATA_DIR, dir_, class_dir)
        for file_name in listdir(class_dir_abs):

            if not file_name.endswith('.jpg'):
                continue

            src_path = join(class_dir_abs, file_name)

            if _is_clean_image(black_list, src_path):
                dest_path = join(dir_cleaned, class_dir, file_name)
            else:
                dest_path = join(dir_junk, class_dir, file_name)

            copyfile(src_path, dest_path)


def _make_labeled_dir_structure(dir_):
    mkdir(dir_)
    for class_ in CLASSES:
        class_dir = join(dir_, str(class_))
        mkdir(class_dir)


def _is_clean_image(black_list, file_path):
    # blacklisted images
    blacklist_format_path = join(
        basename(dirname(dirname(file_path))),
        basename(dirname(file_path)),
        basename(file_path)
    )
    if blacklist_format_path in black_list:
        return False

    # empty images
    if stat(file_path).st_size == 0:
        return False

    # truncated images
    image = Image.open(file_path)
    try:
        image.load()
    except IOError:
        return False

    # green images
    if is_green(img_to_array(image)):
        return False

    return True


if __name__ == '__main__':
    fire.Fire()
