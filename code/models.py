"""
Borrowed from Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import tensorflow as tf
from keras.layers import \
    Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization, ReLU, GlobalAveragePooling2D

import hyperparameters as hp


class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        # TASK 1
        # TODO: Select an optimizer for your network (see the documentation
        #       for tf.keras.optimizers)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = hp.learning_rate)
       #  self.optimizer = tf.keras.optimizers.Adam() #learning_rate = hp.learning_rate)
        # TASK 1
        # TODO: Build your own convolutional neural network, using Dropout at
        #       least once. The input image will be passed through each Keras
        #       layer in self.architecture sequentially. Refer to the imports
        #       to see what Keras layers you can use to build your network.
        #       Feel free to import other layers, but the layers already
        #       imported are enough for this assignment.
        #
        #       Remember: Your network must have under 15 million parameters!
        #       You will see a model summary when you run the program that
        #       displays the total number of parameters of your network.
        #
        #       Remember: Because this is a 15-scene classification task,
        #       the output dimension of the network must be 15. That is,
        #       passing a tensor of shape [batch_size, img_size, img_size, 1]
        #       into the network will produce an output of shape
        #       [batch_size, 15].
        #
        #       Note: Keras layers such as Conv2D and Dense give you the
        #             option of defining an activation function for the layer.
        #             For example, if you wanted ReLU activation on a Conv2D
        #             layer, you'd simply pass the string 'relu' to the
        #             activation parameter when instantiating the layer.
        #             While the choice of what activation functions you use
        #             is up to you, the final layer must use the softmax
        #             activation function so that the output of your network
        #             is a probability distribution.
        #
        #       Note: Flatten is a very useful layer. You shouldn't have to
        #             explicitly reshape any tensors anywhere in your network.

       #prev: (PartialBatchNorm, FullBN)
              # 1.[conv5-10, mp2, fc-32, do-.4, fc-15], 
              # 2.[conv5-10, mp2, conv5-10, mp2, fc-128, do-.4, fc-32, do-.4, fc-15] 
              # 3.[conv7-32, mp2, conv5-32, mp2, 2conv3-64, mp2, fc-256, do-.5, fc-128, do-.5, fc-15] 
              # 4.(PBN) [2conv7-32, mp2, 2conv5-32, mp2, 2conv3-64, mp2, 2conv3-128, mp2, fc-1024, do-.5, fc-1024, do-.5, fc-15] - train = 62.33, test = 54.44
              # 5.(FBN)     [2conv3-32, mp2, 2conv3-64, mp2, 2conv3-128, mp2, 2conv3-256, mp2, 2conv3-256, mp2, fc-2048, do-.5, fc-2048, do-.5, fc-15] - train = 71.40, test=66-70
              # 6.(FBN, PS/V) [2conv3-32, mp3-2, 2conv3-64, mp3-2, 2conv3-128, mp3-2, 3conv3-256, mp3-2, 3conv3-256, mp3-2, fc-2048, do-.5, fc-2048, do-.5, fc-15]  - train = 68, test = 65
              # 7.(FBN, PS) [2conv5-32, mp3-2, 2conv5-64, mp3-2, 2conv3-128, mp3-2, 3conv3-256, mp3-2, 3conv3-256, mp3-2, fc-1024, do-.5, fc-1024, do-.5, fc-15] - train = 72-75, test = 60-70     (MDA)
              # 8.(FBN, PS) [2conv3-32, mp3-2, 2conv3-64, mp3-2, 2conv3-128, mp3-2, 3conv3-256, mp3-2, 3conv3-256, mp3-2, fc-1024, do-.6, fc-1024, do-.6, fc-15] - train = 67-70, test = 55-73 (!) (HDA)
              # 9.(FBN, PS) [2conv3-32, mp3-2, 2conv3-64, mp3-2, 2conv3-128, mp3-2, 3conv3-256, mp3-2, 3conv3-256, mp3-2, fc-1024, do-.5, fc-1024, do-.5, fc-15] - train = , test =                (LDA) [NOT RUN YET]
        self.architecture = [ 
             Conv2D(filters = 32, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
             Conv2D(filters = 32, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
             MaxPool2D(pool_size = (3,3), strides=2),
             
             Conv2D(filters = 64, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
             Conv2D(filters = 64, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
             MaxPool2D(pool_size = (3,3), strides=2),
             
             Conv2D(filters = 128, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
             Conv2D(filters = 128, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
             MaxPool2D(pool_size = (3,3), strides=2),
             
             Conv2D(filters = 256, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
             Conv2D(filters = 256, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
             Conv2D(filters = 256, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
             MaxPool2D(pool_size = (3,3), strides=2),
        
             Conv2D(filters = 256, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
             Conv2D(filters = 256, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
             Conv2D(filters = 256, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
             MaxPool2D(pool_size = (3,3), strides=2),

             Flatten(), 

             Dense(units = 1024), BatchNormalization(), ReLU(),
             Dropout(rate = 0.5),
             Dense(units = 1024), BatchNormalization(), ReLU(),
             Dropout(rate = 0.5),
             Dense(units = 15, activation = 'softmax') #(15, 1)
        ]

    def call(self, x):
        """ Passes input image through the network. """

        for layer in self.architecture:
            x = layer(x)

        return x

    @staticmethod
    def loss_fn(labels, predictions): #labels & predictions are indices -> labels is (n, 1), predictions (n, 15); n = nb inputs
        """ Loss function for the model. """

        # TASK 1
        # TODO: Select a loss function for your network
        #       (see the documentation for tf.keras.losses)
        #
        scce = tf.keras.losses.SparseCategoricalCrossentropy()
        return scce(labels, predictions)

