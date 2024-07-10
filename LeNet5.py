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
LeNet.save("LeNet5.keras")


"""
LeNet-5 is a classic convolutional neural network architecture designed for handwritten digit recognition, 
particularly for the MNIST dataset.

Architecture:
1. Input Layer: Accepts 32x32 pixel grayscale images (MNIST digits are 28x28, zero-padded to 32x32).
2. First Convolutional Layer: Conv2D with 6 filters of size 5x5, followed by AveragePooling2D (2x2).
3. Second Convolutional Layer: Conv2D with 16 filters of size 5x5, followed by AveragePooling2D (2x2).
4. Flatten Layer: Flattens the 2D output of the previous layer into a 1D vector.
5. First Dense Layer: Fully connected with 120 units, using ReLU activation.
6. Second Dense Layer: Fully connected with 84 units, using ReLU activation.
7. Output Layer: Fully connected with 10 units (one for each digit class), using softmax activation.

Key Features:
- LeNet-5 uses convolutional layers for feature extraction and pooling layers for spatial downsampling, 
  which helps in reducing the number of parameters and computation.
- It introduces the concept of convolutional neural networks (CNNs) and demonstrates their effectiveness 
  in image recognition tasks.
- Designed by Yann LeCun in the late 1990s, LeNet-5 laid the foundation for modern CNN architectures 
  and is a milestone in deep learning history.

Usage:
- Suitable for small image classification tasks, especially handwritten digit recognition like MNIST.
- Provides a good introduction to CNNs due to its simple yet effective design principles.

Limitations:
- Limited to grayscale images of fixed size (32x32), which may not be suitable for more complex image datasets 
  or color images.
- May struggle with more intricate patterns or larger datasets compared to modern CNN architectures.

Overall, LeNet-5 remains a fundamental model in the development of deep learning and computer vision, 
showcasing the power of convolutional networks for image classification tasks.
"""
