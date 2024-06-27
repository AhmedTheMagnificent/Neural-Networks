import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train/255
X_test = X_test/255
X_train = X_train.reshape((-1, 28, 28, 1))
X_train = X_train.reshape((-1, 28, 28, 1))
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

LeNet = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=(5,5), activation="relu", input_shape=(28,28,1)),
    tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(filters=16, kernel_size=(5,5), activation="relu"),
    tf.keras.layers.AveragePooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu"),
    tf.keras.layers.Dense(units=84, activation="relu"),
    tf.keras.layers.Dense(units=10, activation="softmax")
])

LeNet.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

LeNet.fit(X_train, y_train, epochs = 10)

LeNet.summary()
loss, acc = LeNet.evaluate(X_test, y_test)
print(f"Accuracy: {acc*100:.2f}%")
LeNet.save("LeNet.keras")

