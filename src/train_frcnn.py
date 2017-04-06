import json
import os
import random
import sys

from keras_frcnn import config

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.setrecursionlimit(40000)

C = config.Config()

from keras_frcnn.simple_parser import get_data

# file_path = sys.argv[1]
roi_file_path = "./../data/roi/roi_bbox.txt"

all_imgs, classes_count, class_mapping = get_data(roi_file_path)

if 'bg' not in classes_count:
    classes_count['bg'] = 0
    class_mapping['bg'] = len(class_mapping)

with open('./../data/roi/classes.json', 'w') as class_data_json:
    json.dump(class_mapping, class_data_json)

inv_map = {v: k for k, v in class_mapping.items()}

# pprint.pprint(classes_count)
print('Num classes (including bg) = {}'.format(len(classes_count)))
random.shuffle(all_imgs)

num_imgs = len(all_imgs)

train_imgs = [s for s in all_imgs if s['imageset'] == 'trainval']
val_imgs = [s for s in all_imgs if s['imageset'] == 'test']

print('Num train samples {}'.format(len(train_imgs)))
print('Num val samples {}'.format(len(val_imgs)))

from keras_frcnn import data_generators
from keras import backend as K

data_gen_train = data_generators.get_anchor_gt(train_imgs, class_mapping, classes_count, C, K.image_dim_ordering(),
                                               mode='train')
data_gen_val = data_generators.get_anchor_gt(val_imgs, class_mapping, classes_count, C, K.image_dim_ordering(),
                                             mode='val')

from keras_frcnn import resnet as nn
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras_frcnn import losses
from keras.callbacks import ReduceLROnPlateau

if K.image_dim_ordering() == 'th':
    input_shape_img = (3, None, None)
else:
    input_shape_img = (None, None, 3)

img_input = Input(shape=input_shape_img)

roi_input = Input(shape=(C.num_rois, 4))

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn = nn.rpn(shared_layers, num_anchors)

# the classifier is build on top of the base layers + the ROI pooling layer + extra layers
classifier = nn.classifier(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count), trainable=True)

# define the full model
model = Model([img_input, roi_input], rpn + classifier)

try:
    print('loading weights from ', C.base_net_weights)
    if os.path.isfile(C.model_path):
        model.load_weights(C.model_path, by_name=True)
    else:
        model.load_weights(C.base_net_weights, by_name=True)

except:
    print('Could not load pretrained model weights. Weights can be found at {} and {}'.format(
        'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_th_dim_ordering_th_kernels_notop.h5',
        'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    ))

optimizer = Adam(lr=0.0001, decay=0.001)
model.compile(optimizer=optimizer,
              loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors), losses.class_loss_cls,
                    losses.class_loss_regr(C.num_rois, len(classes_count) - 1)],
              metrics={'dense_class_{}_loss'.format(len(classes_count)): 'accuracy'})


callbacks = [EarlyStopping(monitor='val_loss', patience=20, verbose=0),
             ModelCheckpoint(C.model_path, monitor='val_loss', save_best_only=True, verbose=0),
             ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1)]

batch_size = 1
nb_epochs = 500
train_samples_per_epoch = len(train_imgs)
nb_val_samples = len(val_imgs)

print('Starting training')

model.fit_generator(data_gen_train, steps_per_epoch=train_samples_per_epoch/batch_size, epochs=nb_epochs,
                    validation_data=data_gen_val, validation_steps=nb_val_samples, callbacks=callbacks,
                    max_q_size=1, workers=1)

# 574/574 [==============================] - 295s - loss: 0.2950 - rpn_out_class_loss: 0.0048 - rpn_out_regress_loss: 0.0366 - dense_class_2_loss: 0.2463 - dense_regress_2_loss: 0.0072 - val_loss: 0.4471 - val_rpn_out_class_loss: 0.0061 - val_rpn_out_regress_loss: 0.0660 - val_dense_class_2_loss: 0.3663 - val_dense_regress_2_loss: 0.0086