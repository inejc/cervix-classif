from functools import partial

import numpy as np
from sklearn.model_selection import cross_val_score


def is_green(image):
    """Detects whether the image is the green thingy. The dataset contains
    truncated images, make sure to handle that properly.
    """
    green = np.array([[[0, 255, 0]]])

    new_image = image - green
    new_image[new_image < 0] = 0

    h, w, _ = new_image.shape
    pixels = h * w

    return np.linalg.norm(new_image) / pixels < 1e-3


def cross_val_scores(classifiers, X, y, scoring='neg_log_loss', k=5):
    cv = partial(
        cross_val_score, X=X, y=y, cv=k, scoring=scoring, n_jobs=-1
    )
    all_clfs_scores = [cv(clf[1]) for clf in classifiers]

    all_mean_clfs_scores = []
    for clf, scores in zip(classifiers, all_clfs_scores):
        mean_score = np.mean(np.abs(scores))
        all_mean_clfs_scores.append((clf[0], mean_score))

    return all_mean_clfs_scores


def create_submission_file(image_names, probs, file_name):
    names_probs = sorted(zip(image_names, probs), key=lambda x: int(x[0][:-4]))

    lines = ["image_name,Type_1,Type_2,Type_3\n"]
    for name, probs in names_probs:
        line = "{:s},{:f},{:f},{:f}\n"
        line = line.format(name, probs[0], probs[1], probs[2])
        lines.append(line)

    with open(file_name, 'w') as f:
        f.writelines(lines)


def read_lines(file_name, line_func=None):
    with open(file_name, 'r') as f:
        if line_func is None:
            return list(f)
        else:
            return [line_func(l) for l in list(f)]
