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
        # self.architecture = [ 
        #      Conv2D(filters = 32, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
        #      Conv2D(filters = 32, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
        #      MaxPool2D(pool_size = (3,3), strides=2),
             
        #      Conv2D(filters = 64, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
        #      Conv2D(filters = 64, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
        #      MaxPool2D(pool_size = (3,3), strides=2),
             
        #      Conv2D(filters = 128, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
        #      Conv2D(filters = 128, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
        #      MaxPool2D(pool_size = (3,3), strides=2),
             
        #      Conv2D(filters = 256, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
        #      Conv2D(filters = 256, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
        #      Conv2D(filters = 256, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
        #      MaxPool2D(pool_size = (3,3), strides=2),
        
        #      Conv2D(filters = 256, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
        #      Conv2D(filters = 256, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
        #      Conv2D(filters = 256, kernel_size = (3,3), padding='same'), BatchNormalization(), ReLU(),
        #      MaxPool2D(pool_size = (3,3), strides=2),

        #      Flatten(),
         

        #      Dense(units = 1024), BatchNormalization(), ReLU(),
        #      Dropout(rate = 0.5),
        #      Dense(units = 1024), BatchNormalization(), ReLU(),
        #      Dropout(rate = 0.5),
        #      Dense(units = 15, activation = 'softmax') #(15, 1)
        # ]


        self.conv_blocks = [
           # Block 1
           Conv2D(64, 3, padding="same", activation=None, name="block1_conv1"),
           BatchNormalization(name="block1_bn1"),
           ReLU(name="block1_relu1"),
           Conv2D(64, 3, padding="same", activation=None, name="block1_conv2"),
           BatchNormalization(name="block1_bn2"),
           ReLU(name="block1_relu2"),
           MaxPool2D(2, name="block1_pool"),
           Dropout(0.3, name="block1_dropout"),
          
           # Block 2
           Conv2D(128, 3, padding="same", activation=None, name="block2_conv1"),
           BatchNormalization(name="block2_bn1"),
           ReLU(name="block2_relu1"),
           Conv2D(128, 3, padding="same", activation=None, name="block2_conv2"),
           BatchNormalization(name="block2_bn2"),
           ReLU(name="block2_relu2"),
           MaxPool2D(2, name="block2_pool"),
           Dropout(0.3, name="block2_dropout"),
          
           # Block 3
           Conv2D(256, 3, padding="same", activation=None, name="block3_conv1"),
           BatchNormalization(name="block3_bn1"),
           ReLU(name="block3_relu1"),
           Conv2D(256, 3, padding="same", activation=None, name="block3_conv2"),
           BatchNormalization(name="block3_bn2"),
           ReLU(name="block3_relu2"),
           MaxPool2D(2, name="block3_pool"),
           Dropout(0.3, name="block3_dropout"),
          
           # Block 4
           Conv2D(512, 3, padding="same", activation=None, name="block4_conv1"),
           BatchNormalization(name="block4_bn1"),
           ReLU(name="block4_relu1"),
           Conv2D(512, 3, padding="same", activation=None, name="block4_conv2"),
           BatchNormalization(name="block4_bn2"),
           ReLU(name="block4_relu2"),
           MaxPool2D(2, name="block4_pool"),
           Dropout(0.3, name="block4_dropout"),
       ]


       # Fully Connected Layers
        self.head = [
           GlobalAveragePooling2D(name="global_avg_pool"),
           Dense(512, activation="relu", name="fc1"),
           Dropout(0.3, name="dropout1"),
           Dense(512, activation="relu", name="fc2"),
           Dropout(0.3, name="dropout2"),
           Dense(1, activation="sigmoid", name="output")  # Binary classification
       ]


       # Convert the convolutional blocks and head into sequential models
        self.conv_blocks = tf.keras.Sequential(self.conv_blocks, name="conv_base")
        self.head = tf.keras.Sequential(self.head, name="head")


    def call(self, x):
        """ Passes the input through the network. """
        x = self.conv_blocks(x)
        x = self.head(x)
        return x


    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for binary classification. """
        return tf.keras.losses.BinaryCrossentropy()(labels, predictions)

