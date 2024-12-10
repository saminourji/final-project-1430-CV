import os
import sys
import argparse
import re
from datetime import datetime
import tensorflow as tf

import hyperparameters as hp
from models import YourModel
from preprocess import Datasets
from skimage.transform import resize
from tensorboard_utils import ImageLabelingLogger, CustomModelSaver
from matplotlib import pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def subsample_large_dataset(data_iterator, step=100):
    """ Subsamples a DirectoryIterator by selecting every nth item. """
    # Load all data and labels into memory
    images, labels = [], []
    for i in range(len(data_iterator)):
        x, y = data_iterator[i]  # Get batch
        images.append(x)
        labels.append(y)

    # Concatenate into arrays
    images = np.concatenate(images)
    labels = np.concatenate(labels)

    # Subsample the arrays
    images = images[::step]
    labels = labels[::step]

    # Create a new data generator from the subsampled data
    return tf.keras.preprocessing.image.ImageDataGenerator().flow(images, labels, batch_size=hp.batch_size)

def train(model, datasets, checkpoint_path, logs_path, init_epoch):
    """ Training routine. """

    # Subsample train and test data
    datasets.train_data = subsample_large_dataset(datasets.train_data, step=100)
    datasets.test_data = subsample_large_dataset(datasets.test_data, step=100)

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            update_freq='batch',
            profile_batch=0),
        ImageLabelingLogger(logs_path, datasets),
        CustomModelSaver(checkpoint_path, "1", hp.max_num_weights)
    ]

    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=hp.num_epochs,
        batch_size=None,  # Required as None as we use an ImageDataGenerator; see preprocess.py get_data()
        callbacks=callback_list,
        initial_epoch=init_epoch,
    )


def main():
    """ Main function. """

    path = "../data"

    print("Kaggle path: ", path)
    time_now = datetime.now()
    timestamp = time_now.strftime("%m%d%y-%H%M%S")
    init_epoch = 0

    os.chdir(sys.path[0])

    datasets = Datasets(path, "1")
    
    model = YourModel()
    model(tf.keras.Input(shape=(hp.img_size, hp.img_size, 3)))
    checkpoint_path = "checkpoints" + os.sep + \
        "your_model" + os.sep + timestamp + os.sep
    logs_path = "logs" + os.sep + "your_model" + \
        os.sep + timestamp + os.sep

    # Print summary of model
    model.summary()

    # Make checkpoint directory if needed
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    # Compile model graph
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["accuracy"])

    train(model, datasets, checkpoint_path, logs_path, init_epoch)

main()
