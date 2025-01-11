import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from PIL import Image

#CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0

#rgb to gray for dataset
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#training and test images to grayscale
gray_train = np.expand_dims(rgb2gray(X_train), axis=-1)
gray_test = np.expand_dims(rgb2gray(X_test), axis=-1)

#U-Net model
def build_unet(input_shape):
    inputs = Input(shape=input_shape)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    deconv1 = Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(conv3)
    deconv1 = concatenate([deconv1, conv2], axis=-1)
    deconv1 = Conv2D(128, (3, 3), activation='relu', padding='same')(deconv1)
    deconv2 = Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(deconv1)
    deconv2 = concatenate([deconv2, conv1], axis=-1)
    deconv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(deconv2)
    outputs = Conv2D(3, (1, 1), activation='sigmoid', padding='same')(deconv2)
    model = Model(inputs, outputs)
    return model

input_shape = gray_train.shape[1:]
model = build_unet(input_shape)
model.compile(optimizer='adam', loss='mean_squared_error')
# model.summary()

checkpoint = ModelCheckpoint('unet_colorization_weights.keras', save_best_only=True, monitor='val_loss', mode='min')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

#train
history = model.fit(gray_train, X_train, validation_data=(gray_test, X_test), epochs=40, batch_size=64,
                    callbacks=[checkpoint, early_stopping])

# Save
model.save('unet_colorization_model2.keras')

# Predict
predicted_color = model.predict(gray_test)
