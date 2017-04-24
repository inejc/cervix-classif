import glob
import json
import os
import sys

import cv2
import fire
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from tqdm import tqdm

import keras_frcnn.resnet as nn
from data_provider import DATA_DIR, ROI_CLASSES_FILE, FRCNN_MODELS_DIR
from keras_frcnn.config import Config
from keras_frcnn.roi_helpers import non_max_suppression_fast, apply_regr, rpn_to_roi
from keras_frcnn.roi_helpers import resize_bounding_box

sys.setrecursionlimit(40000)


def format_img(img, C):
    img_min_side = C.im_size
    img, new_height, new_width = resize_image(img, img_min_side)
    img = img[:, :, (2, 1, 0)]
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    img = np.expand_dims(img, axis=0)

    img[:, 0, :, :] -= C.mean_pixel[0]
    img[:, 1, :, :] -= C.mean_pixel[1]
    img[:, 2, :, :] -= C.mean_pixel[2]
    return img, new_width, new_height


def resize_image(img, img_min_side):
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
    return img, new_height, new_width


def get_class_mappings():
    with open(ROI_CLASSES_FILE, 'r') as class_data_json:
        class_mapping = json.load(class_data_json)
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)
    return {v: k for k, v in class_mapping.items()}


def get_model_rpn(input_shape_img, C):
    img_input = Input(shape=input_shape_img)
    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input)
    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn = nn.rpn(shared_layers, num_anchors)
    model_rpn = Model(img_input, rpn + [shared_layers])
    model_rpn.load_weights(C.get_model_path(), by_name=True)
    model_rpn.compile(optimizer='adam', loss='mse')
    return model_rpn


def get_model_classifier(class_mapping, input_shape_features, C):
    feature_map_input = Input(shape=input_shape_features)
    roi_input = Input(shape=(C.num_rois, 4))
    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping))
    model_classifier = Model([feature_map_input, roi_input], classifier)
    model_classifier.load_weights(C.get_model_path(), by_name=True)
    model_classifier.compile(optimizer='adam', loss='mse')
    return model_classifier


def load_config(model_name):
    model_dir = os.path.join(FRCNN_MODELS_DIR, model_name)
    with open(os.path.join(model_dir, 'config.json'), 'r') as file:
        # json.dump(config, file, default=lambda o: o.__dict__, indent=4, separators=(',', ': '))
        return json.load(file)


def predict(model_name, in_dir="train_cleaned", bbox_threshold=0.5):
    C = Config(**load_config(model_name))
    C.use_horizontal_flips = False
    C.use_vertical_flips = False

    class_mapping = get_class_mappings()

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
        input_shape_features = (1024, None, None)
    else:
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, 1024)

    model_rpn = get_model_rpn(input_shape_img, C)
    model_classifier = get_model_classifier(class_mapping, input_shape_features, C)

    images = sorted(glob.glob(os.path.join(DATA_DIR, in_dir, "**/*.jpg"), recursive=True))
    print("Found " + str(len(images)) + " images...")

    probs = []
    boxes = []
    for idx, img_name in tqdm(enumerate(images), total=len(images)):
        img = cv2.imread(img_name)
        height, width, _ = img.shape
        X, new_width, new_height = format_img(img, C)

        if K.image_dim_ordering() == 'tf':
            X = np.transpose(X, (0, 2, 3, 1))
        # get the feature maps and output from the RPN
        [Y1, Y2, F] = model_rpn.predict(X)

        R = rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(), overlap_thresh=0.7)

        # convert from (x1,y1,x2,y2) to (x,y,w,h)
        R[:, 2] = R[:, 2] - R[:, 0]
        R[:, 3] = R[:, 3] - R[:, 1]

        # apply the spatial pyramid pooling to the proposed regions
        bboxes = {}
        boxes.append({})
        probs.append({})
        for jk in range(R.shape[0] // C.num_rois + 1):
            ROIs = np.expand_dims(R[C.num_rois * jk:C.num_rois * (jk + 1), :], axis=0)
            if ROIs.shape[1] == 0:
                break

            if jk == R.shape[0] // C.num_rois:
                # pad R
                curr_shape = ROIs.shape
                target_shape = (curr_shape[0], C.num_rois, curr_shape[2])
                ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
                ROIs_padded[:, :curr_shape[1], :] = ROIs
                ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
                ROIs = ROIs_padded

            [P_cls, P_regr] = model_classifier.predict([F, ROIs])
            P_regr = P_regr / C.std_scaling

            for ii in range(P_cls.shape[1]):
                if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
                    continue

                cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]
                if cls_name not in bboxes:
                    bboxes[cls_name] = []
                    boxes[idx][cls_name] = []
                    probs[idx][cls_name] = []

                (x, y, w, h) = ROIs[0, ii, :]

                cls_num = np.argmax(P_cls[0, ii, :])
                (tx, ty, tw, th) = P_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                tx /= C.classifier_regr_std[0]
                ty /= C.classifier_regr_std[1]
                tw /= C.classifier_regr_std[2]
                th /= C.classifier_regr_std[3]
                x, y, w, h = apply_regr(x, y, w, h, tx, ty, tw, th)

                bboxes[cls_name].append([16 * x, 16 * y, 16 * (x + w), 16 * (y + h)])
                probs[idx][cls_name].append(np.max(P_cls[0, ii, :]))

        for key in bboxes:
            bbox = np.array(bboxes[key])
            boxes[idx][key] = [resize_bounding_box(width / new_width, height / new_height, b) for b in bbox]

    np.savez_compressed(
        file=os.path.join(FRCNN_MODELS_DIR, model_name, in_dir + "_predictions"),
        images=images,
        boxes=boxes,
        probs=probs,
    )


def crop(model_name, in_dir, overlap_th=0.95):
    images, boxes, probs = load_predictions(model_name, in_dir)

    print("Found " + str(len(images)) + " images...")
    for idx, img_name in tqdm(enumerate(images), total=len(images)):

        out_dir_name = img_name.split(DATA_DIR)[1].split("/")[1]
        new_image_path = img_name.replace(out_dir_name, out_dir_name + "_frcnn_cropped")

        img = cv2.imread(img_name)
        bbox = np.array(boxes[idx].get("cervix"))

        if bbox is None or len(bbox) == 0:
            print("Could not find ROI on image " + img_name)
            cv2.imwrite(new_image_path, img)
            continue

        new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[idx]["cervix"]), overlap_thresh=overlap_th)
        (x1, y1, x2, y2) = new_boxes[np.argmax(new_probs), :]
        cv2.imwrite(new_image_path, img[y1:y2, x1:x2])


def visualize(model_name, in_dir, only_best=True, overlap_th=0.95, img_min_side=600):
    images, boxes, probs = load_predictions(model_name, in_dir)

    print("Found " + str(len(images)) + " images...")
    for idx, img_name in tqdm(enumerate(images), total=len(images)):

        img = cv2.imread(img_name)

        for key in boxes[idx]:
            bbox = np.array(boxes[idx][key])
            new_boxes, new_probs = non_max_suppression_fast(bbox, np.array(probs[idx][key]), overlap_thresh=overlap_th)

            if only_best:
                x1, y1, x2, y2 = new_boxes[np.argmax(new_probs), :]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
                continue

            for jk in range(new_boxes.shape[0]):
                x1, y1, x2, y2 = new_boxes[jk, :]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)

                text_label = '{}:{}'.format(key, int(100 * new_probs[jk]))
                (retval, baseLine) = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                text_org = (x1, y1 + 20)
                cv2.rectangle(img, (text_org[0] - 5, text_org[1] + baseLine - 5),
                              (text_org[0] + retval[0] + 5, text_org[1] - retval[1] - 5), (0, 0, 0), 5)
                cv2.putText(img, text_label, text_org, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

        img, _, _ = resize_image(img, img_min_side)
        cv2.imshow('img', img)
        cv2.waitKey(0)


def load_predictions(model_name, in_dir):
    file = os.path.join(FRCNN_MODELS_DIR, model_name, in_dir + "_predictions.npz")
    if not os.path.isfile(file):
        raise RuntimeError("ROI_PREDICTIONS_FILE not found! You must run predict first!")
    with np.load(file) as data:
        return data["images"], data["boxes"], data["probs"]


if __name__ == '__main__':
    fire.Fire()
    # predict("neki", in_dir="test")
    visualize("neki", in_dir="test", overlap_th=0.7, only_best=False)
