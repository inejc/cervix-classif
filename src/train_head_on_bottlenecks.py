from os.path import join

import h5py
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2

from data_provider import MODELS_DIR


def fit(epochs=1000, lr=0.001, dropout=0.5, l2_reg=0.05):
    WEIGHTS_PATH = join(MODELS_DIR, 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
    BOTTLENECKS_FILE = join(MODELS_DIR, "resnet50_bottlenecks.h5")
    LABELS_FILE = join(MODELS_DIR, "resnet50_labels.h5")

    MODEL_PATH = join(MODELS_DIR, "resnet50_head_model_on_bottlenecks.h5")

    with h5py.File(BOTTLENECKS_FILE) as hf:
        X_train = hf["train"][:]
        X_valid = hf["valid"][:]

    with h5py.File(LABELS_FILE) as hf:
        y_train = hf["train"][:]
        y_valid = hf["valid"][:]

    cb = [ModelCheckpoint(MODEL_PATH, save_best_only=True)]

    model = Sequential([
        Dropout(dropout, input_shape=X_train.shape[1:]),
        # Dense(8, init='uniform', activation='relu'),
        # Dense(4, init='uniform', activation='relu'),
        Dense(3, activation="softmax", W_regularizer=l2(l2_reg))
    ])

    model.compile(Adam(lr=lr, decay=0.01), "categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train, y_train, callbacks=cb, validation_data=(X_valid, y_valid), epochs=epochs)

    print("Finished fitting the model!")
