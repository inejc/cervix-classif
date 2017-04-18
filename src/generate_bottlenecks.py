import time
from os.path import join

import h5py
from keras.layers import GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from data_provider import load_organized_data_info, MODELS_DIR
from resnet50 import ResNet50
from math import ceil

WEIGHTS_PATH = join(MODELS_DIR, 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
BOTTLENECKS_FILE = join(MODELS_DIR, "resnet50_bottlenecks.h5")
LABELS_FILE = join(MODELS_DIR, "resnet50_labels.h5")

data_info = load_organized_data_info(imgs_dim=299, name="cropped_tsmole_path")

dir_tr, num_tr = data_info['dir_tr'], data_info['num_tr']
dir_val, num_val = data_info['dir_val'], data_info['num_val']
dir_te, num_te = data_info['dir_te'], data_info['num_te']

batch_size = 32

body = ResNet50(input_shape=(299, 299, 3),
                weights_path=WEIGHTS_PATH)
head = body.output
head = GlobalAveragePooling2D()(head)
model = Model(body.input, head)

gen = ImageDataGenerator(vertical_flip=True,
                         zoom_range=0.05,
                         fill_mode="constant",
                         channel_shift_range=10,
                         rotation_range=5,
                         width_shift_range=0.05,
                         height_shift_range=0.05)

train_batches = gen.flow_from_directory(dir_tr, model.input_shape[1:3], shuffle=False,
                                        batch_size=batch_size)
valid_batches = gen.flow_from_directory(dir_val, model.input_shape[1:3], shuffle=False,
                                        batch_size=batch_size)
test_batches = gen.flow_from_directory(dir_te, model.input_shape[1:3], shuffle=False,
                                       batch_size=batch_size, class_mode=None)

start_time = time.time()
train_bottleneck = model.predict_generator(train_batches, ceil(num_tr / batch_size))
print("--- train bottleneck %s seconds ---" % (time.time() - start_time))

start_time = time.time()
valid_bottleneck = model.predict_generator(valid_batches, ceil(num_val / batch_size))
print("--- valid bottleneck  %s seconds ---" % (time.time() - start_time))

start_time = time.time()
test_bottleneck = model.predict_generator(test_batches, ceil(num_te / batch_size))
print("--- test bottleneck %s seconds ---" % (time.time() - start_time))

start_time = time.time()
with h5py.File(BOTTLENECKS_FILE) as hf:
    hf.create_dataset("train", data=train_bottleneck)
    hf.create_dataset("valid", data=valid_bottleneck)
    hf.create_dataset("test", data=test_bottleneck)
print("--- bottlenecks dataset %s seconds ---" % (time.time() - start_time))

start_time = time.time()
with h5py.File(LABELS_FILE) as hf:
    hf.create_dataset("train", data=to_categorical(train_batches.classes))
    hf.create_dataset("valid", data=to_categorical(valid_batches.classes))
print("--- label dataset %s seconds ---" % (time.time() - start_time))