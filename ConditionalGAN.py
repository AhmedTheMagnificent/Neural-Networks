import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train/255
X_train = X_train.reshape((-1, 28, 28, 1))
y_train = tf.keras.utils.to_categorical(y_train, 10)


generator = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_dim=110),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(28 * 28 * 1, activation="sigmoid"),
    tf.keras.layers.Reshape((28, 28, 1))
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Inputs(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, kernel_size=5, padding="same", strides=2),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(128, kernel_size=5, padding="same", strides=2),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(512, kernel_size=5, padding="same", strides=2),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    tf.keras.layers.Dropout(0.5).
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])