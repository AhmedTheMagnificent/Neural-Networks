import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

baseModel = tf.keras.applications.MobileNetV2(
    weights=None,
    include_top=False,
    input_shape=(32, 32, 3)
)

x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
output = tf.keras.layers.Dense(10, activation='softmax')(x)

MobileNetV2 = tf.keras.models.Model(inputs=baseModel.input, outputs=output)

MobileNetV2.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])

MobileNetV2.fit(X_train, y_train, epochs=10, batch_size=64)

MobileNetV2.summary()
loss, acc = MobileNetV2.evaluate(X_test, y_test)
print(f"Accuracy: {acc*100:.2f}%")
MobileNetV2.save("MobileNetV2.keras")


"""
MobileNetV2 is a lightweight convolutional neural network architecture designed for efficient inference 
in mobile and embedded vision applications.

Architecture:
1. Input Layer: Accepts 32x32 pixel RGB images (CIFAR-10 dataset).
2. Base Model (MobileNetV2): 
   - Initial Convolution Layer: 32 filters of size 3x3, stride 2.
   - Bottleneck Layers: 19 bottleneck layers with varying configurations of filters, kernel sizes, and strides.
     * Bottleneck blocks use depthwise separable convolutions, which consist of:
       a. Depthwise Convolution: Applies a single convolutional filter per input channel (depth).
       b. Pointwise Convolution: Uses a 1x1 convolution to combine the outputs of the depthwise convolution.
     * Inverted Residuals: Shortcut connections between thin bottleneck layers.
     * Linear Bottlenecks: Prevent activation functions from being applied to the bottleneck's output.
   - Configuration of Bottleneck Layers:
     - 1 layer, 16 filters, stride 1
     - 2 layers, 24 filters, stride 2
     - 3 layers, 32 filters, stride 2
     - 4 layers, 64 filters, stride 2
     - 3 layers, 96 filters, stride 1
     - 3 layers, 160 filters, stride 2
     - 1 layer, 320 filters, stride 1
3. Global Average Pooling Layer: Reduces the spatial dimensions of the feature maps to a single value per feature map.
4. Dense Layer: Fully connected layer with 512 units and ReLU activation.
5. Output Layer: Fully connected layer with 10 units (one for each CIFAR-10 class) and softmax activation.

Key Features:
- Efficiency: MobileNetV2 is designed to be computationally efficient, making it suitable for mobile and embedded applications.
- Depthwise Separable Convolutions: These convolutions significantly reduce the number of parameters and computation compared to standard convolutions.
- Inverted Residuals: Introduce shortcut connections between thin bottleneck layers, improving the efficiency and performance.
- Modularity: The architecture can be easily scaled up or down by adjusting the number of layers or the width multiplier.

Usage:
- Suitable for image classification tasks, particularly on datasets like CIFAR-10.
- Ideal for deployment in resource-constrained environments like mobile devices and embedded systems.

Limitations:
- May not achieve the same level of accuracy as larger, more complex models on some tasks.
- Designed primarily for speed and efficiency, so there may be trade-offs in terms of performance on very large or complex datasets.
"""