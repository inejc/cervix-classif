import json
import os
import random
import sys
import time

import fire
import numpy as np
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from data_provider import ROI_CLASSES_FILE, ROI_BBOX_FILE
from keras_frcnn import config, roi_helpers
from keras_frcnn import data_generators
from keras_frcnn import resnet as nn
from keras_frcnn.losses import rpn_loss_cls, rpn_loss_regr, class_loss_regr, class_loss_cls
from keras_frcnn.simple_parser import get_data
from model_utils import dump_args, LoggingCallback

sys.setrecursionlimit(40000)
random.seed(0)

C = config.Config()


def build_model(classes_count, num_anchors):
    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)
    # define the RPN, built on the base layers
    rpn = nn.rpn(shared_layers, num_anchors)
    # the classifier is build on top of the base layers + the ROI pooling layer + extra layers
    classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count),
                               trainable=True)
    # define the full model
    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)
    try:
        print('loading weights from ', C.base_net_weights)
        if os.path.isfile(C.get_model_path()):
            model_rpn.load_weights(C.get_model_path(), by_name=True)
            model_classifier.load_weights(C.get_model_path(), by_name=True)
        else:
            model_rpn.load_weights(C.base_net_weights, by_name=True)
            model_classifier.load_weights(C.base_net_weights, by_name=True)
    except:
        print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
            'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        ))
        exit()
    return model_rpn, model_classifier, model_all


# TODO: AUTO SAVE MODELS PARAMS AND LINK THEM TO NAME!
# TODO: SAVE FIT HISTORY/SUMMARY


@dump_args
def train(model_name, epochs=60, lr=0.0001, decay=0.001):
    all_imgs, classes_count, class_mapping = get_data(ROI_BBOX_FILE)
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)

    C.model_name = model_name
    print("Model name: " + C.model_name)

    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    if not os.path.isfile(ROI_CLASSES_FILE):
        with open(ROI_CLASSES_FILE, 'w') as class_data_json:
            json.dump(class_mapping, class_data_json)

    print('Num classes (including bg) = {}'.format(len(classes_count)))
    random.shuffle(all_imgs)

    train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

    print('Num train samples {}'.format(len(train_imgs)))
    print('Num val samples {}'.format(len(val_imgs)))

    model_rpn, model_classifier, model_all = build_model(classes_count, num_anchors)

    optimizer = Adam(lr=lr, decay=decay)
    optimizer_classifier = Adam(lr=1e-7)

    model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[class_loss_cls, class_loss_regr(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    data_gen_train = data_generators.get_anchor_gt(train_imgs, classes_count, C, K.image_dim_ordering(), mode='train')
    data_gen_val = data_generators.get_anchor_gt(val_imgs, classes_count, C, K.image_dim_ordering(), mode='val')

    # callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=0),
    #              ModelCheckpoint(C.get_model_path(), monitor='val_loss', save_best_only=True, verbose=0),
    #              ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    #              LoggingCallback(C)]

    epoch_length = 10
    iter_num = 0
    epoch_num = 0

    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()

    best_loss = np.Inf

    print('Starting training')
    while True:
        X, Y, img_data = next(data_gen_train)
        loss_rpn = model_rpn.train_on_batch(X, Y)
        P_rpn = model_rpn.predict_on_batch(X)
        R = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7,
                                   max_boxes=300)

        # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
        X2, Y1, Y2 = roi_helpers.calc_iou(R, img_data, C, class_mapping)

        neg_samples = np.where(Y1[0, :, -1] == 1)[0]
        pos_samples = np.where(Y1[0, :, -1] == 0)[0]

        rpn_accuracy_rpn_monitor.append(len(pos_samples))
        rpn_accuracy_for_epoch.append((len(pos_samples)))

        if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
            mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
            rpn_accuracy_rpn_monitor = []
            print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                mean_overlapping_bboxes, epoch_length))
            if mean_overlapping_bboxes == 0:
                print(
                    'RPN is not producing bounding boxes that overlap the ground truth boxes. Results will not be satisfactory. Keep training.')

        if X2 is None or X2.shape[1] < C.num_rois:
            continue

        if C.num_rois > 1:
            if len(pos_samples) < C.num_rois / 2:
                selected_pos_samples = pos_samples.tolist()
            else:
                selected_pos_samples = np.random.choice(pos_samples, C.num_rois / 2, replace=False).tolist()

            selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples),
                                                    replace=False).tolist()
            sel_samples = selected_pos_samples + selected_neg_samples
        else:
            # in the extreme case where num_rois = 1, we pick a random pos or neg sample
            selected_pos_samples = pos_samples.tolist()
            selected_neg_samples = neg_samples.tolist()
            if np.random.randint(0, 2):
                sel_samples = random.choice(selected_neg_samples)
            else:
                sel_samples = random.choice(selected_pos_samples)

        P = model_classifier.predict([X, X2[:, sel_samples, :]])
        loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]],
                                                     [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])

        losses[iter_num, 0] = loss_rpn[1]
        losses[iter_num, 1] = loss_rpn[2]

        losses[iter_num, 2] = loss_class[1]
        losses[iter_num, 3] = loss_class[2]
        losses[iter_num, 4] = loss_class[3]

        iter_num += 1

        if iter_num == epoch_length:
            loss_rpn_cls = np.mean(losses[:, 0])
            loss_rpn_regr = np.mean(losses[:, 1])
            loss_class_cls = np.mean(losses[:, 2])
            loss_class_regr = np.mean(losses[:, 3])
            class_acc = np.mean(losses[:, 4])

            mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
            rpn_accuracy_for_epoch = []

            if C.verbose:
                print('Epoch {}:'.format(epoch_num))
                print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                    mean_overlapping_bboxes))
                print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                print('Loss RPN classifier: {}'.format((loss_rpn_cls)))
                print('Loss RPN regression: {}'.format((loss_rpn_regr)))
                print('Loss Classifier classifier: {}'.format((loss_class_cls)))
                print('Loss Classifier regression: {}'.format((loss_class_regr)))
                print('Elapsed time: {}'.format(time.time() - start_time))
            else:
                print(
                    'loss_rpn_cls,{},loss_rpn_regr,{},loss_class_cls,{},loss_class_regr,{},class_acc,{},elapsed_time,{}'.format(
                        loss_rpn_cls, loss_rpn_regr, loss_class_cls, loss_class_regr, class_acc,
                        time.time() - start_time))
            curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
            iter_num = 0
            start_time = time.time()
            epoch_num += 1
            if epoch_num == 1 or curr_loss < best_loss:
                if C.verbose:
                    print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                best_loss = curr_loss
                model_all.save_weights(C.get_model_path())
        if epoch_num == epochs:
            print('Training complete, exiting.')
            sys.exit()


if __name__ == '__main__':
    fire.Fire()