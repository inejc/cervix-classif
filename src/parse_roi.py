import glob

import ijroi
import cv2
import numpy as np

from keras_frcnn.simple_parser import get_data
from keras_frcnn.data_generators import get_new_img_size
from keras_frcnn.data_augment import augment

roi_file_path = "./../data/roi/roi_bbox.txt"
mean_pixel_file_path = './../data/roi/mean_pixel_color.txt'
roi_files_dir = './../data/roi/train/*/*.roi'


def process_roi():
    generate_roi_file()
    generate_mean_pixel_file()


def generate_mean_pixel_file():
    from keras_frcnn.config import Config
    C = Config()
    all_imgs, _, _ = get_data(roi_file_path)

    avg = [0, 0, 0]
    for img_data in all_imgs:
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
    np.savetxt(mean_pixel_file_path, avg, delimiter=',')


def generate_roi_file():
    roi_files = glob.glob(roi_files_dir)
    with open(roi_file_path, 'w') as out:
        for roi_file in roi_files:
            with open(roi_file, "rb") as f:
                roi = ijroi.read_roi(f)
                out.write(roi_file.replace("roi/", "").replace(".roi", ".jpg") + ", ")
                out.write(", ".join(map(str, roi[0][::-1])) + ", ")
                out.write(", ".join(map(str, roi[2][::-1])) + ", ")
                out.write("cervix\n")


def get_average_roi_size():
    # TODO TIM
    with open(roi_file_path, 'r') as f:
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
    process_roi()
