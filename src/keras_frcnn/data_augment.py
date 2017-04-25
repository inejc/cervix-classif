import copy

import cv2
import numpy as np
from scipy.ndimage import zoom

from keras_frcnn import roi_helpers


def augment(img_data, config, augment=True):
    assert 'filepath' in img_data
    assert 'bboxes' in img_data
    assert 'width' in img_data
    assert 'height' in img_data

    img_data_aug = copy.deepcopy(img_data)

    img = cv2.imread(img_data_aug['filepath'])

    # BGR -> RGB
    img = img[:, :, (2, 1, 0)]

    if augment:
        height, width = img.shape[:2]

        if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 1)
            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                bbox['x2'] = width - x1
                bbox['x1'] = width - x2

        if config.use_vertical_flips and np.random.randint(0, 2) == 0:
            img = cv2.flip(img, 0)
            for bbox in img_data_aug['bboxes']:
                y1 = bbox['y1']
                y2 = bbox['y2']
                bbox['y2'] = height - y1
                bbox['y1'] = height - y2

        if config.random_rotate:
            angle = np.random.choice([0, 90, 180, 270], 1)[0]
            if angle == 270:
                img = np.transpose(img, (1, 0, 2))
                img = cv2.flip(img, 0)
            elif angle == 180:
                img = cv2.flip(img, -1)
            elif angle == 90:
                img = np.transpose(img, (1, 0, 2))
                img = cv2.flip(img, 1)
            elif angle == 0:
                pass

            for bbox in img_data_aug['bboxes']:
                x1 = bbox['x1']
                x2 = bbox['x2']
                y1 = bbox['y1']
                y2 = bbox['y2']
                if angle == 270:
                    bbox['x1'] = y1
                    bbox['x2'] = y2
                    bbox['y1'] = width - x2
                    bbox['y2'] = width - x1
                elif angle == 180:
                    bbox['x2'] = width - x1
                    bbox['x1'] = width - x2
                    bbox['y2'] = height - y1
                    bbox['y1'] = height - y2
                elif angle == 90:
                    bbox['x1'] = height - y2
                    bbox['x2'] = height - y1
                    bbox['y1'] = x1
                    bbox['y2'] = x2
                elif angle == 0:
                    pass

        if config.scale_augment and np.random.randint(0, 2) == 0:
            scale = config.scale_percent
            scale = np.random.uniform(low=min(1, scale), high=max(1, scale))
            img = clipped_zoom(img, scale)
            for bbox in img_data_aug['bboxes']:
                coordinates = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                coordinates = roi_helpers.resize_bounding_box(scale, scale, coordinates)
                bbox['x1'] = coordinates[0]
                bbox['y1'] = coordinates[1]
                bbox['x2'] = coordinates[2]
                bbox['y2'] = coordinates[3]

    return img_data_aug, img


def clipped_zoom(img, zoom_factor, **kwargs):
    h, w = img.shape[:2]

    # width and height of the zoomed image
    zh = int(np.round(zoom_factor * h))
    zw = int(np.round(zoom_factor * w))

    # for multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # zooming out
    if zoom_factor < 1:
        # bounding box of the clip region within the output array
        top = (h - zh) // 2
        left = (w - zw) // 2
        # zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)

    # zooming in
    elif zoom_factor > 1:
        # bounding box of the clip region within the input array
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # if zoom_factor == 1, just return the input array
    else:
        out = img
    return out
