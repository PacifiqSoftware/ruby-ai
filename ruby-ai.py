# Importing libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from colorama import Fore, Back, Style
import time
import os

os.system("cls")
time.sleep(4)
os.system("cls")

print(Fore.RED + """
╭━━━┳╮╱╭┳━━┳╮╱╱╭╮
┃╭━╮┃┃╱┃┃╭╮┃╰╮╭╯┃
┃╰━╯┃┃╱┃┃╰╯╰╮╰╯╭╯
┃╭╮╭┫┃╱┃┃╭━╮┣╮╭╯
┃┃┃╰┫╰━╯┃╰━╯┃┃┃
╰╯╰━┻━━━┻━━━╯╰╯
© PacificSoftware, 2023-present""")
print(Fore.WHITE + """""")

# Define the generator model
def build_generator(z_dim):
    model = Sequential()

    model.add(Dense(256, input_dim=z_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(28*28*1, activation='tanh'))
    model.add(Reshape((28, 28, 1)))

    z = Input(shape=(z_dim,))
    img = model(z)

    return Model(z, img)

# Define the discriminator model
def build_discriminator(img_shape):
    model = Sequential()

    model.add(Flatten(input_shape=img_shape))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

# Define the combined model (stacked generator and discriminator)
def build_gan(generator, discriminator):
    discriminator.trainable = False

    z = Input(shape=(z_dim,))
    img = generator(z)
    validity = discriminator(img)

    return Model(z, validity)

# Load and preprocess the dataset (MNIST)
(X_train, _), (_, _) = mnist.load_data()

# Rescale -1 to 1
X_train = X_train / 127.5 - 1.
X_train = np.expand_dims(X_train, axis=3)

# Define the input shape
img_shape = X_train.shape[1:]

# Define the latent dimension
z_dim = 100

# Build and compile the discriminator
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy',
                      optimizer=Adam(0.0002, 0.5),
                      metrics=['accuracy'])

# Build the generator
generator = build_generator(z_dim)

# Build and compile the GAN model
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Train the GAN model
epochs = 10000
batch_size = 32

for epoch in range(epochs):

    # Generate a random noise sample
    z = np.random.normal(0, 1, (batch_size, z_dim))

    # Generate images from the noise sample using the generator
    gen_imgs = generator.predict(z)

    # Select a random batch of real images from the dataset
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    # Train the discriminator on real and generated images
    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size,)))

    # Train the generator-discriminator stack on the noise data. Labels are flipped to train the generator.
    g_loss = gan.train_on_batch(z, np.zeros((batch_size, 1)))

    # Print the progress and save generated samples
    if epoch % 1000 == 0:
        print(f"Epoch {epoch} | D loss: {d_loss[0]:.4f}, D accuracy: {d_loss[1]:.2f} | G loss: {g_loss:.4f}")
        save_imgs(generator, epoch)

