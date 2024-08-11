import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

(X_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
X_train = X_train / 255.0
X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_train = tf.reshape(X_train, (X_train.shape[0], 28, 28, 1))

generator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(100,)),
    tf.keras.layers.Dense(7 * 7 * 128, activation="relu"),
    tf.keras.layers.Reshape((7, 7, 128)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(128, kernel_size=5, padding="same", strides=2, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2DTranspose(1, kernel_size=5, padding="same", strides=2, activation="sigmoid")
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(64, kernel_size=5, padding="same", strides=2),
    tf.keras.layers.LeakyReLU(negative_slope=0.2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(128, kernel_size=5, padding="same", strides=2),
    tf.keras.layers.LeakyReLU(negative_slope=0.2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Conv2D(256, kernel_size=5, padding="same", strides=2),
    tf.keras.layers.LeakyReLU(negative_slope=0.2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

class FashionGAN(tf.keras.models.Model):
    def __init__(self, generator, discriminator):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

    def compile(self):
        super().compile()

    def train(self, batch):
        realImages = batch
        noise = tf.random.normal([realImages.shape[0], 100])
        fakeImages = self.generator(noise, training=False)

        with tf.GradientTape() as dTape:
            yhatReal = self.discriminator(realImages, training=True)
            yhatFake = self.discriminator(fakeImages, training=True)
            yhatRealFake = tf.concat([yhatReal, yhatFake], axis=0)
            yRealFake = tf.concat([tf.ones_like(yhatReal), tf.zeros_like(yhatFake)], axis=0)
            noiseReal = 0.15 * tf.random.normal(tf.shape(yhatReal))
            noiseFake = -0.15 * tf.random.normal(tf.shape(yhatFake))
            yRealFake += tf.concat([noiseReal, noiseFake], axis=0)
            dLoss = tf.keras.losses.BinaryCrossentropy()(yRealFake, yhatRealFake)

        dGrad = dTape.gradient(dLoss, self.discriminator.trainable_variables)
        tf.keras.optimizers.Adam(learning_rate=0.00001).apply_gradients(zip(dGrad, self.discriminator.trainable_variables))

        with tf.GradientTape() as gTape:
            noise = tf.random.normal([realImages.shape[0], 100])
            generatedImages = self.generator(noise, training=True)
            predictedLabels = self.discriminator(generatedImages, training=True)
            gLoss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(predictedLabels), predictedLabels)

        gGrad = gTape.gradient(gLoss, self.generator.trainable_variables)
        tf.keras.optimizers.Adam(learning_rate=0.0001).apply_gradients(zip(gGrad, self.generator.trainable_variables))

        return {"gLoss": gLoss, "dLoss": dLoss}

GAN = FashionGAN(generator, discriminator)
GAN.compile()

history = {'gLoss': [], 'dLoss': []}
epochs = 30
batchSize = 256

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    for i in range(0, X_train.shape[0], batchSize):
        print(f"Batch {i // batchSize + 1}/{X_train.shape[0] // batchSize}")
        batch = X_train[i:i + batchSize]
        losses = GAN.train(batch)
        history['gLoss'].append(losses['gLoss'])
        history['dLoss'].append(losses['dLoss'])

generator.save("generatorModelFinal.keras")
discriminator.save("discriminatorModelFinal.keras")

plt.suptitle('Loss')
plt.plot(history['dLoss'], label='dLoss')
plt.plot(history['gLoss'], label='gLoss')
plt.legend()
plt.show()

path = "generatorModelFinal.keras"

if os.path.exists(path):
    generator = tf.keras.models.load_model(path)
    print("Model loaded successfully.")
else:
    print(f"File not found: {path}")

def generateImages(generator, numImages=5, noiseDim=100):
    noise = tf.random.normal([numImages, noiseDim])
    generatedImages = generator(noise, training=False)
    generatedImages = 0.5 * generatedImages + 0.5

    plt.figure(figsize=(10, 2))
    for i in range(numImages):
        plt.subplot(1, numImages, i + 1)
        plt.imshow(generatedImages[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

if os.path.exists(path):
    generateImages(generator, numImages=5)
else:
    print("Unable to generate images, model not loaded.")
