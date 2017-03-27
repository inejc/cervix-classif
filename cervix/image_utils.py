import numpy as np


def is_green(image, show_image=False):
    """Detect whether the image is the green thingy.

    The dataset contains truncated images, make sure to handle that properly.

    """
    green = np.array([[[0, 255, 0]]])

    new_image = image - green
    new_image[new_image < 0] = 0

    h, w, _ = new_image.shape
    pixels = h * w

    return np.linalg.norm(new_image) / pixels < 1e-3