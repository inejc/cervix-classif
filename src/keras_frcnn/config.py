import os

import numpy as np
from keras import backend as K


class Config:
    def __init__(self):
        # setting for roi augmentation
        self.use_horizontal_flips = True
        self.use_vertical_flips = False
        self.scale_augment = True
        self.scale_bounds = [0.]
        self.random_rotate = True
        self.random_rotate_scale = 60.

        # anchor box scales
        # self.anchor_box_scales = [128, 256, 512]
        self.anchor_box_scales = [50, 100, 150, 200, 299]

        # anchor box ratios
        self.anchor_box_ratios = [[1, 1], [1, 2], [2, 1]]

        # size to resize the smallest side of the image
        self.im_size = 299

        # number of ROIs at once
        self.num_rois = 2

        # stride at the RPN (this depends on the network configuration)
        self.rpn_stride = 16

        self.balanced_classes = False

        # scaling the stdev
        self.std_scaling = 4.0

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        if os.path.isfile('./../data/roi/mean_pixel_color.txt'):
            self.mean_pixel = np.loadtxt('./../data/roi/mean_pixel_color.txt', delimiter=',')

        # location of pretrained weights for the base network
        # weight files can be found at:
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5
        # https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
        if K.image_dim_ordering() == 'th':
            self.base_net_weights = './../models/resnet50_weights_th_dim_ordering_th_kernels_notop.h5'
        else:
            self.base_net_weights = './../models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        self.model_path = './../models/model_frcnn.hdf5'
