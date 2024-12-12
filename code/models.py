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

    def __init__(self, fourier):
        super(YourModel, self).__init__()
        print("Fourier:", self.fourier)
        self.fourier = fourier
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = hp.learning_rate)
        
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
        if self.fourier:
            self.head = [] #flattenning done inside of the code
        else: 
            self.head = [GlobalAveragePooling2D(name="global_avg_pool")]
        self.head += [
           Dense(512, activation="relu", name="fc1"),
           Dropout(0.3, name="dropout1"),
           Dense(512, activation="relu", name="fc2"),
           Dropout(0.3, name="dropout2"),
           Dense(1, activation="sigmoid", name="output")  # Binary classification
       ]


       # Convert the convolutional blocks and head into sequential models
        self.conv_blocks = tf.keras.Sequential(self.conv_blocks, name="conv_base")
        self.head = tf.keras.Sequential(self.head, name="head")

    def apply_fourier_transform(self, x):
        """ Applies Fourier Transform to the input tensor. """
        x = tf.cast(x, tf.float32)  # Ensure input is float32
        x = tf_signal.rfft2d(x)  # Apply real FFTi
        x_mag = tf.abs(x)  # Compute magnitude
        x_phase = tf.math.angle(x)  # Compute phase
        print(x_mag.shape)
        print(x_phase.shape)
        return x_mag, x_phase

    fourier = True
    def call(self, x):
        if self.fourier:
            """ Passes the input through the network. """
            x_mag, x_phase  = self.apply_fourier_transform(x)

            # Pass the original input through convolutional blocks
            conv_output = self.conv_blocks(x)

            # Flatten the outputs for concatenation
            conv_output_flattened = tf.keras.layers.Flatten()(conv_output)
            x_mag_flattened = tf.keras.layers.Flatten()(x_mag)  # Flatten magnitude
            x_phase_flattened = tf.keras.layers.Flatten()(x_phase)  # Flatten phase

            # Concatenate the Fourier Transform with the convolutional block output
            combined_features = tf.keras.layers.Concatenate()([conv_output_flattened, x_mag_flattened, x_phase_flattened])

            # Pass the combined features through the head layers
            x = self.head(combined_features)

        else:
            x = self.conv_blocks(x)
            x = self.head(x)
        return x


    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for binary classification. """
        return tf.keras.losses.BinaryCrossentropy()(labels, predictions)

