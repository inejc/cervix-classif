from os.path import join

import fire
import numpy as np

from data_provider import SUBMISSIONS_DIR
from utils import read_lines, create_submission_file

W_SUBMISSIONS = {
    'xception_fine_tuned_cleaned_0_72837.csv': 6,
    'xception_fine_tuned_cleaned_0_73764.csv': 5,
    'xception_fine_tuned_cleaned_0_74836.csv': 4,
    'xception_fine_tuned_0.76231.csv': 2,
    'xception_fine_tuned_0_77085.csv': 1,
    'xception_fine_tuned_cleaned_0_77236.csv': 1,
}


def average():
    names, probs, weights = [], [], []

    for file_name, weight in W_SUBMISSIONS.items():
        file_path = join(SUBMISSIONS_DIR, file_name)
        lines = read_lines(file_path)[1:]

        weights.append(weight)

        single_probs = []
        names = []

        for line in lines:
            split = line.rstrip().split(',')
            names.append(split[0])
            single_probs.append(np.array([float(x) for x in split[1:]]))

        probs.append(np.array(single_probs))

    probs = np.array(probs)
    weights = np.array(weights)

    averaged = probs * weights[:, np.newaxis, np.newaxis]
    averaged = np.sum(averaged, axis=0) / np.sum(weights)

    submissions_file = join(SUBMISSIONS_DIR, 'averaged.csv')
    create_submission_file(names, averaged, submissions_file)


if __name__ == '__main__':
    fire.Fire()
