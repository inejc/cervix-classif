import glob
import json
import os
import sys

import cv2
import numpy as np

from keras_frcnn import config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.setrecursionlimit(40000)
C = config.Config()
C.use_horizontal_flips = False
C.use_vertical_flips = False


def format_img(img):
    img_min_side = C.im_size
    (height, width, _) = img.shape

    if width <= height:
        f = img_min_side / width
        new_height = int(f * height)
        new_width = int(img_min_side)
    else:
        f = img_min_side / height
        new_width = int(f * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img = img[:, :, (2, 1, 0)]
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img[:, 0, :, :] -= C.mean_pixel[0]  # used to be 103.939
    img[:, 1, :, :] -= C.mean_pixel[1]  # used to be 116.779
    img[:, 2, :, :] -= C.mean_pixel[2]  # used to be 123.68
    return img, new_width, new_height


def resize_bounding_box(width_ratio, height_ratio, coordinates):
    """coordinates: (x1, x2, y1, y2) """
    x1 = max(0, coordinates[0] * width_ratio)
    x2 = max(0, coordinates[1] * width_ratio)
    y1 = max(0, coordinates[2] * height_ratio)
    y2 = max(0, coordinates[3] * height_ratio)
    return int(x1), int(x2), int(y1), int(y2)


with open('./../data/roi/classes.json', 'r') as class_data_json:
    class_mapping = json.load(class_data_json)

if 'bg' not in class_mapping:
    class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.items()}

class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
num_rois = C.num_rois

import keras_frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
    input_shape_features = (1024, None, None)
else:
    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)

img_input = Input(shape=input_shape_img)

feature_map_input = Input(shape=input_shape_features)

roi_input = Input(shape=(num_rois, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

# classifier, uses base layers + proposals
print(class_mapping)

classifier = nn.classifier(feature_map_input, roi_input, num_rois, nb_classes=len(class_mapping))

model_rpn = Model(img_input, rpn + [shared_layers])
model_classifier = Model([feature_map_input, roi_input], classifier)

model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

visualise = False

print('Parsing annotation files')
# img_path = sys.argv[1]
# FIXME TIM
img_path = "./../data/train/Type_3/"

for idx, img_name in enumerate(sorted(glob.glob(os.path.join(img_path, '*.jpg')))):

    img = cv2.imread(img_name)
    height, width, _ = img.shape

    X, new_width, new_height = format_img(img)

    width_ratio = width / new_width
    height_ratio = height / new_height

    img_scaled = np.transpose(X[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
    img_scaled[:, :, 0] += C.mean_pixel[2]
    img_scaled[:, :, 1] += C.mean_pixel[1]
    img_scaled[:, :, 2] += C.mean_pixel[0]

    img_scaled = img_scaled.astype(np.uint8)

    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))
    # get the feature maps and output from the RPN
    [Y1, Y2, F] = model_rpn.predict(X)

    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering())

    # convert from (x1,y1,x2,y2) to (x,y,w,h)
    R[:, 2] = R[:, 2] - R[:, 0]
    R[:, 3] = R[:, 3] - R[:, 1]

    # apply the spatial pyramid pooling to the proposed regions
    bboxes = {}
    probs = {}
    for jk in range(R.shape[0] // num_rois + 1):
        ROIs = np.expand_dims(R[num_rois * jk:num_rois * (jk + 1), :], axis=0)
        if ROIs.shape[1] == 0:
            break

        if jk == R.shape[0] // num_rois:
            # pad R
            curr_shape = ROIs.shape
            target_shape = (curr_shape[0], num_rois, curr_shape[2])
            ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
            ROIs_padded[:, :curr_shape[1], :] = ROIs
            ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
            ROIs = ROIs_padded

        [P_cls, P_regr] = model_classifier.predict([F, ROIs])
        P_regr = P_regr / C.std_scaling

        for ii in range(P_cls.shape[1]):

            # FIXME TIM:  only one bounding box with max P_cls
            if np.max(P_cls[0, ii, :]) < 0.5 or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                continue

            cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

            if cls_name not in bboxes:
                bboxes[cls_name] = []
                probs[cls_name] = []

            (x, y, w, h) = ROIs[0, ii, :]

            cls_num = np.argmax(P_cls[0, ii, :])
            (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
            x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)

            bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
            probs[cls_name].append(np.max(P_cls[0, ii, :]))

    all_dets = {}

    best_match = None

    for key in bboxes:
        bbox = np.array(bboxes[key])

        # TODO TIM: change overlapTrashold
        new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]),
                                                                    overlapThresh=0.9)

        # TODO TIM: check if box makes sense
        best_match = new_boxes[np.argmax(new_probs), :]
        best_match = resize_bounding_box(width_ratio, height_ratio, best_match)
        if visualise:
            (x1, y1, x2, y2) = best_match
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            for jk in range(new_boxes.shape[0]):
                x1, x2, y1, y2 = resize_bounding_box(width_ratio, height_ratio, new_boxes[jk, :])

                # FIXME TIM: class_to_color[key] namesto hardcodane barve
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

                textLabel = '{}:{}'.format(key, int(100 * new_probs[jk]))
                if key not in all_dets:
                    all_dets[key] = 100 * new_probs[jk]
                else:
                    all_dets[key] = max(all_dets[key], 100 * new_probs[jk])

                (retval, baseLine) = cv2.getTextSize(textLabel, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                textOrg = (x1, y1 + 20)

                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5), (0, 0, 0),
                              2)
                cv2.rectangle(img, (textOrg[0] - 5, textOrg[1] + baseLine - 5),
                              (textOrg[0] + retval[0] + 5, textOrg[1] - retval[1] - 5),
                              (255, 255, 255), -1)
                cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                            1)

    if best_match is not None:
        (x1, y1, x2, y2) = best_match
        cv2.imwrite(img_name.replace("/train", "/cropped/train"), img[y1:y2, x1:x2])
    else:
        print("Could not find ROI on image " + img_name)
        cv2.imwrite(img_name.replace("/train", "/cropped/train"), img)

    if visualise and best_match is not None:
        cv2.imshow('img', img_scaled)
        cv2.waitKey(0)
