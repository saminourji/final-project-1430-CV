import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPool2D, Dropout, Flatten, Dense, GlobalAveragePooling2D, Concatenate
import tensorflow.signal as tf_signal

class YourModel(tf.keras.Model):
    """ Neural network model with Fourier Transform layer. """

    def __init__(self):
        super(YourModel, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # Convolutional Blocks
        self.conv_blocks = [
            Conv2D(64, 3, padding="same", activation=None, name="block1_conv1"),
            BatchNormalization(name="block1_bn1"),
            ReLU(name="block1_relu1"),
            Conv2D(64, 3, padding="same", activation=None, name="block1_conv2"),
            BatchNormalization(name="block1_bn2"),
            ReLU(name="block1_relu2"),
            MaxPool2D(2, name="block1_pool"),
            Dropout(0.3, name="block1_dropout"),

            Conv2D(128, 3, padding="same", activation=None, name="block2_conv1"),
            BatchNormalization(name="block2_bn1"),
            ReLU(name="block2_relu1"),
            Conv2D(128, 3, padding="same", activation=None, name="block2_conv2"),
            BatchNormalization(name="block2_bn2"),
            ReLU(name="block2_relu2"),
            MaxPool2D(2, name="block2_pool"),
            Dropout(0.3, name="block2_dropout"),

            Conv2D(256, 3, padding="same", activation=None, name="block3_conv1"),
            BatchNormalization(name="block3_bn1"),
            ReLU(name="block3_relu1"),
            Conv2D(256, 3, padding="same", activation=None, name="block3_conv2"),
            BatchNormalization(name="block3_bn2"),
            ReLU(name="block3_relu2"),
            MaxPool2D(2, name="block3_pool"),
            Dropout(0.3, name="block3_dropout"),

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
            Dense(512, activation="relu", name="fc1"),
            Dropout(0.3, name="dropout1"),
            Dense(512, activation="relu", name="fc2"),
            Dropout(0.3, name="dropout2"),
            Dense(1, activation="sigmoid", name="output")  # Binary classification
        ]

        self.conv_blocks = tf.keras.Sequential(self.conv_blocks, name="conv_base")
        self.head = tf.keras.Sequential(self.head, name="head")

    def apply_fourier_transform(self, x):
        """ Applies Fourier Transform to the input tensor. """
        x = tf.cast(x, tf.float32)  # Ensure input is float32
        x = tf_signal.rfft2d(x)  # Apply RFFT
        x = tf.abs(x)  # Take magnitude of Fourier coefficients
        x = tf.reduce_mean(x, axis=-1, keepdims=True)  # Reduce along frequency dimensions
        return x

    def call(self, x):
        """ Passes the input through the network. """
        # Compute Fourier Transform of the input
        fourier_transform = self.apply_fourier_transform(x)

        # Pass the original input through convolutional blocks
        conv_output = self.conv_blocks(x)

        # Flatten the outputs for concatenation
        conv_output_flattened = tf.keras.layers.Flatten()(conv_output)
        fourier_transform_flattened = tf.keras.layers.Flatten()(fourier_transform)

        # Concatenate the Fourier Transform with the convolutional block output
        combined_features = tf.keras.layers.Concatenate()([conv_output_flattened, fourier_transform_flattened])

        # Pass the combined features through the head layers
        x = self.head(combined_features)
        return x

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for binary classification. """
        return tf.keras.losses.BinaryCrossentropy()(labels, predictions)
    