from collections import OrderedDict
from os import listdir
from os.path import join, splitext

import numpy as np
import ijroi
import fire
import roi
from keras.preprocessing.image import load_img

CLASSES = ['Type_1', 'Type_2', 'Type_3']


def _get_dict_roi(directory):
    d = OrderedDict()
    for class_ in CLASSES:
        for f in listdir(join(directory, class_)):
            img_id = splitext(f)[0].split('_')[2]
            d[img_id] = join(directory, class_, f)
    return d


def _get_dict_all_images(directory):
    d = OrderedDict()
    for class_ in CLASSES:
        for f in listdir(join(directory, class_)):
            img_id = splitext(f)[0]
            d[img_id] = join(directory, class_, f)
    return d


def _convert_from_roi(fname):
    """Convert a roi file to a numpy array [x, y, h, w].

    Parameters
    ----------
    fname : string
        If ends with `.roi`, we assume a full path is given

    """
    with open(fname, 'rb') as f:
        roi = ijroi.read_roi(f)
        top, left = roi[0]
        bottom, right = roi[2]
        height, width = bottom - top, right - left

        return np.array([top, left, height, width])


def resize_roi_to_original(image_dir, roi_dir, output_dir, initial_size):
    roi_dict = _get_dict_roi(roi_dir)
    img_dict = _get_dict_all_images(image_dir)

    for img_id in roi_dict:
        img = load_img(img_dict[img_id])
        bounding_box = _convert_from_roi(roi_dict[img_id]).astype(np.float64)
        bounding_box /= initial_size
        bounding_box[[0, 2]] *= img.height
        bounding_box[[1, 3]] *= img.width
        bounding_box = bounding_box.astype(np.int32)

        new_file = join(output_dir, img_dict[img_id].split('/')[-2], '%s.roi' % img_id)
        roi.save_prediction(bounding_box, new_file)


if __name__ == '__main__':
    fire.Fire()