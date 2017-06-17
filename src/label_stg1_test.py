from os import mkdir, rename
from os.path import join

import fire

from data_provider import DATA_DIR


def label():
    val_dir = join(DATA_DIR, 'val_299_final')
    mkdir(join(val_dir, 'Type_1'))
    mkdir(join(val_dir, 'Type_2'))
    mkdir(join(val_dir, 'Type_3'))

    labels_file = join(DATA_DIR, 'solution_stg1_release.csv')
    with open(labels_file) as labels:
        def split_(x):
            s = x.strip().split(',')
            return s[0], s[1:].index('1') + 1
        names_labels = [split_(x) for x in list(labels)[1:]]

    for name, label_ in names_labels:
        src = join(val_dir, name)
        dst = join(val_dir, 'Type_{:d}'.format(label_), name)
        rename(src, dst)


if __name__ == '__main__':
    fire.Fire()
