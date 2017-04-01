import numpy as np


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


def create_submission_file(image_names, probs, file_name):
    names_probs = sorted(zip(image_names, probs), key=lambda x: int(x[0][:-4]))

    lines = ["image_name,Type_1,Type_2,Type_3\n"]
    for name, probs in names_probs:
        line = "{:s},{:f},{:f},{:f}\n"
        line = line.format(name, probs[0], probs[1], probs[2])
        lines.append(line)

    with open(file_name, 'w') as f:

        f.writelines(lines)
