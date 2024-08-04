import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()
X_train = X_train/255
X_train = np.expand_dims(X_train, axis=-1)
y_train = tf.keras.utils.to_categorical(y_train, 10)
print(X_train.shape)