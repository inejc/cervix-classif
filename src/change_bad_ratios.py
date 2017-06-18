from os import listdir
from os.path import join
from shutil import copy

import fire
from PIL.Image import open

from data_provider import DATA_DIR


def change():
    dir_src = join(DATA_DIR, 'test_stg2')
    dir_dst = join(DATA_DIR, 'test_stg2_frcnn_cropped')

    for file in listdir(dir_dst):
        file_path = join(dir_dst, file)

        image = open(file_path)
        width, height = image.size
        ratio = height / width if width > height else width / height

        if ratio < 0.5:
            copy(join(dir_src, file), file_path)


if __name__ == '__main__':
    fire.Fire()
