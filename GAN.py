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
batch_size = 256

for epoch in range(epochs):
    for i in range(0, X_train.shape[0], batch_size):
        batch = X_train[i:i + batch_size]
        losses = GAN.train(batch)
        history['gLoss'].append(losses['gLoss'])
        history['dLoss'].append(losses['dLoss'])

generator.save("generator_model_final.keras")
discriminator.save("discriminator_model_final.keras")

plt.suptitle('Loss')
plt.plot(history['dLoss'], label='dLoss')
plt.plot(history['gLoss'], label='gLoss')
plt.legend()
plt.show()

path = ("generator_model_final.keras")

if os.path.exists(path):
    generator = tf.keras.models.load_model(path)

def generate_images(generator, num_images=5, noise_dim=100):
    noise = tf.random.normal([num_images, noise_dim])
    generated_images = generator(noise, training=False)
    generated_images = 0.5 * generated_images + 0.5

    plt.figure(figsize=(10, 2))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.show()

if os.path.exists(path):
    generate_images(generator, num_images=5)

"""
FashionGAN is a Generative Adversarial Network (GAN) designed to generate images that resemble those in the Fashion MNIST dataset. The GAN consists of two main components: a Generator and a Discriminator.

Architecture:

1. Generator:
   - Input Layer: A 100-dimensional noise vector.
   - Dense Layer: Fully connected layer that reshapes the input into a 7x7x128 tensor.
   - Conv2DTranspose Layers: Three transposed convolutional layers with ReLU activation and batch normalization. The final layer outputs a 28x28x1 image with sigmoid activation.

2. Discriminator:
   - Input Layer: A 28x28x1 grayscale image.
   - Conv2D Layers: Three convolutional layers with Leaky ReLU activation and dropout for regularization. Each layer reduces the spatial dimensions of the input.
   - Flatten Layer: Flattens the output into a 1D vector.
   - Dense Layer: Fully connected layer with sigmoid activation to classify the input as real or fake.

Key Features:
- The Generator is trained to produce images that can deceive the Discriminator.
- The Discriminator is trained to differentiate between real and fake images.
- The model uses Binary Cross-Entropy as the loss function for both the Generator and Discriminator, optimizing with the Adam optimizer.

Usage:
- The script first trains the GAN on the Fashion MNIST dataset.
- After training, it saves the Generator and Discriminator models.
- The saved Generator model is then used to generate new images from random noise, showcasing the GAN's ability to create fashion-related images.

Limitations:
- The model may struggle to generate high-quality images if not trained for a sufficient number of epochs or with an appropriate batch size.
- GAN training can be unstable, requiring careful tuning of hyperparameters like learning rate and noise addition.
"""
