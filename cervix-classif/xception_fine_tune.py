"""
Usage:
    python xception_fine_tune.py train
"""

import fire

from data_provider import load_organized_data_info

HEIGHT, WIDTH = 299, 299


def train():
    data_info = load_organized_data_info(imgs_dim=HEIGHT)
    print(data_info)


if __name__ == '__main__':
    fire.Fire()
