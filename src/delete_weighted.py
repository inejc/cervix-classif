from os import listdir, remove
from os.path import join

import fire

from data_provider import DATA_DIR


def delete(soft=True):
    dir_ = join(DATA_DIR, 'train_299_final')

    weighted = [x for x in filter(lambda x: 'weighted' in x, listdir(join(dir_, 'Type_2')))]
    for file in weighted[:140]:
        print(file)
        if not soft:
            remove(join(dir_, 'Type_2', file))

    weighted = [x for x in filter(lambda x: 'weighted' in x, listdir(join(dir_, 'Type_3')))]
    for file in weighted[:135]:
        print(file)
        if not soft:
            remove(join(dir_, 'Type_3', file))


if __name__ == '__main__':
    fire.Fire()
