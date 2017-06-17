from os import remove
from os.path import join

import fire

from data_provider import DATA_DIR


def remove_():
    tr_dir = join(DATA_DIR, 'train_299_final')
    file_name_blueprint = 'additional_cleaned_frcnn_cropped_{:s}'

    with open('test_stg1_additional_duplicates.csv') as lines:
        def split_(x):
            s = x.strip().split(',')
            s = s[1].split('/')
            return join(tr_dir, s[1], file_name_blueprint.format(s[2]))
        files = [split_(x) for x in list(lines)[1:]]
        files = list(set(files))

    for file in files:
        remove(file)


if __name__ == '__main__':
    fire.Fire()
