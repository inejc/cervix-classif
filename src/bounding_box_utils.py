from collections import OrderedDict, Counter
from os import listdir
from os.path import join, splitext, isdir
from pprint import pprint

import fire
import ijroi
import numpy as np
import roi
from keras.preprocessing.image import load_img, img_to_array

CLASSES = ['Type_1', 'Type_2', 'Type_3']

# Since the large majority of the images have aspect ratio 4:3, we will resize
# them accordingly
HEIGHT, WIDTH = 400, 300


def _get_dict_roi(directory):
    d = OrderedDict()
    for class_ in CLASSES:
        for f in listdir(join(directory, class_)):
            img_id = splitext(f)[0]
            d[img_id] = join(directory, class_, f)
    return d


def _get_dict_all_images(directory):
    d = OrderedDict()
    for class_ in CLASSES:
        for f in listdir(join(directory, class_)):
            img_id = splitext(f)[0]
            d[img_id] = join(directory, class_, f)
    return d


def _get_dict_tagged_images(directory, roi_directory):
    """Get all available images in the training directory.

    Returns
    -------
    dict : {<image_id>: <image file path>}

    """
    all_images = _get_dict_all_images(directory)
    tagged_roi = _get_dict_roi(roi_directory)
    d = OrderedDict()
    for img_id in all_images:
        if img_id in tagged_roi:
            d[img_id] = all_images[img_id]
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


def get_all_images(directory):
    img_dict = _get_dict_all_images(directory)
    X = np.zeros((len(img_dict), HEIGHT, WIDTH, 3))
    for idx, img_id in enumerate(img_dict):
        X[idx] = load_img(img_dict[img_id])
    return list(img_dict.keys()), X


def get_tagged_images(image_dir, roi_dir):
    roi_dict = _get_dict_roi(roi_dir)
    img_dict = _get_dict_tagged_images(image_dir, roi_dir)
    # Initialize X and Y (contains 4 values x, y, w, h)
    X = np.zeros((len(img_dict), HEIGHT, WIDTH, 3))
    Y = np.zeros((len(img_dict), 4))
    # Load the image files into a nice data array
    for idx, key in enumerate(img_dict):
        img = load_img(img_dict[key], target_size=(HEIGHT, WIDTH))
        X[idx] = img_to_array(img)
        Y[idx] = _convert_from_roi(roi_dict[key])

    return list(img_dict.keys()), X, Y


def resize_roi_to_original(image_dir, roi_dir, output_dir, initial_size):
    roi_dict = _get_dict_roi(roi_dir)
    img_dict = _get_dict_all_images(image_dir)

    for img_id in roi_dict:
        img = load_img(img_dict[img_id])

        bounding_box = _convert_from_roi(roi_dict[img_id]).astype(np.float64)
        bounding_box /= initial_size

        img_size = np.array(img.size)
        sq_img_size = np.array([img_size.min(), img_size.min()])

        # Calculate the amount of padding needed on each size of the new
        # bounding box
        img_padding = (img_size - sq_img_size) / 2
        img_padding = np.hstack((img_padding[::-1], [0, 0]))

        bounding_box *= sq_img_size[0]
        bounding_box = bounding_box.astype(np.int32)

        # Add the padding to the bounding box
        bounding_box = bounding_box + img_padding

        new_file = join(output_dir, img_dict[img_id].split('/')[-2], '%s.roi' % img_id)
        roi.save_prediction(bounding_box, new_file)


def image_info(image_dir):
    sizes, aspect_ratios = [], []
    for f in listdir(image_dir):
        try:
            img = load_img(join(image_dir, f))
            sizes.append(img.size)
            aspect_ratios.append(img.width / img.height)
        except:
            print('Could not open `%s`' % join(image_dir, f))
    for class_ in CLASSES:
        if not isdir(join(image_dir, class_)):
            continue
        for f in listdir(join(image_dir, class_)):
            try:
                img = load_img(join(image_dir, class_, f))
                sizes.append(img.size)
                aspect_ratios.append(img.width / img.height)
            except:
                print('Could not open `%s`' % join(image_dir, class_, f))

    sizes = Counter(sizes)
    aspect_ratios = Counter(aspect_ratios)

    print('Resolutions:')
    pprint(sizes)
    print('Aspect ratios:')
    pprint(aspect_ratios)


if __name__ == '__main__':
    fire.Fire()
