import tensorflow as tf
import numpy as np

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train/255
X_test = X_test/255
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

baseModel = tf.keras.applications.ResNet50(
    weights=None,
    include_top=False,
    input_shape=(32, 32, 3)
)

x = baseModel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(512, activation="relu")(x)
output = tf.keras.layers.Dense(10, activation="softmax")(x)

ResNet50 = tf.keras.models.Model(baseModel.input, output)

ResNet50.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=['accuracy'])

ResNet50.fit(X_train, y_train, epochs = 10)

ResNet50.summary()
loss, acc = ResNet50.evaluate(X_test, y_test)
print(f"Accuracy: {acc*100:.2f}%")
ResNet50.save("ResNet-50.keras")