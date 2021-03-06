import glob

import os
import cv2
import ijroi
import numpy as np

from data_provider import ROI_DIR, MEAN_PIXEL_FILE, ROI_BBOX_FILE
from keras_frcnn.config import Config
from keras_frcnn.data_augment import augment
from keras_frcnn.data_generators import get_new_img_size
from keras_frcnn.simple_parser import get_data


def process_roi():
    generate_roi_file()
    generate_mean_pixel_file()


def generate_mean_pixel_file():
    C = Config()
    all_imgs, _, _ = get_data(ROI_BBOX_FILE)

    avg = [0, 0, 0]
    for img_data in all_imgs:
        print(img_data['filepath'])
        img_data_aug, x_img = augment(img_data, C, augment=False)

        (width, height) = (img_data_aug['width'], img_data_aug['height'])
        (rows, cols, _) = x_img.shape

        # get image dimensions for resizing
        (resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

        # resize the image so that smalles side is length = 600px
        x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
        pixels = (resized_width * resized_height)
        avg[0] += np.sum(x_img[:, :, 0]) / pixels
        avg[1] += np.sum(x_img[:, :, 1]) / pixels
        avg[2] += np.sum(x_img[:, :, 2]) / pixels
    avg = [a / len(all_imgs) for a in list(avg)]
    np.savetxt(MEAN_PIXEL_FILE, avg, delimiter=',')


def generate_roi_file():
    roi_files = glob.glob(os.path.join(ROI_DIR, '**/*.roi', recursive=True))
    with open(ROI_BBOX_FILE, 'w') as out:
        for roi_file in roi_files:
            with open(roi_file, "rb") as f:
                roi = ijroi.read_roi(f)
                out.write(roi_file.replace("roi/", "").replace(".roi", ".jpg") + ", ")
                out.write(", ".join(map(str, roi[0][::-1])) + ", ")
                out.write(", ".join(map(str, roi[2][::-1])) + ", ")
                out.write("cervix\n")


def get_average_roi_size():
    # TODO TIM
    with open(ROI_BBOX_FILE, 'r') as f:
        width = []
        height = []
        ratios = []
        for line in f.readlines():
            line = line.split(",")
            w = abs(int(line[1]) - int(line[3]))
            width.append(w)
            h = abs(int(line[2]) - int(line[4]))
            height.append(h)
            ratios.append(w / h)

        print(np.mean(width))
        print(np.mean(height))
        print(np.mean(ratios))


if __name__ == '__main__':
    import fire

    fire.Fire()
