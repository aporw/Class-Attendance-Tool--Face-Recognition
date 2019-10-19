import keras
from keras.layers import Input, add
import numpy as np
from keras.models import Model
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.utils import np_utils

import cv2

dirct = "D:\Project\Face_Recongnition\FaceRec-master_Image\FaceRec-master\One_Pic"
features=cv2.imread(dirct+"\\ankur4.JPG")
input_shape = features.shape

input_features = features.reshape(-1,input_shape[0], input_shape[1],input_shape[2])

x = Input(shape=(input_shape[0], input_shape[1],input_shape[2])) 

# Encoder
conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
conv1_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
pool2 = MaxPooling2D((2, 2), padding='same')(conv1_2)
conv1_3 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool2)
pool3 = MaxPooling2D((2, 2), padding='same')(conv1_3)
print(pool3.shape)
conv1_4 = Conv2D(4, (3, 3), activation='relu', padding='same')(pool3)
h = MaxPooling2D((1, 1), padding='same')(conv1_4)
#print(h.shape)

# Decoder
conv2_0 = Conv2D(4, (3, 3), activation='relu', padding='same')(h)
up0 = UpSampling2D((1, 1))(conv2_0)
print(up0.shape)
conv2_1 = Conv2D(8, (3, 3), activation='relu', padding='same')(up0)
up1 = UpSampling2D((2, 2))(conv2_1)
conv2_2 = Conv2D(8, (3, 3), activation='relu', padding='same')(up1)
up2 = UpSampling2D((2, 2))(conv2_2)
conv2_3 = Conv2D(16, (3, 3), activation='relu')(up2)
up3 = UpSampling2D((2, 2))(conv2_3)
r = Conv2D(input_shape[2], (3, 3), activation='linear', padding='same')(up3)

autoencoder = Model(inputs=x, outputs=r)
autoencoder.compile(optimizer='adadelta', loss='mse')


epochs =25
batch_size = 1

history = autoencoder.fit(input_features, input_features, batch_size=batch_size, epochs=epochs, verbose=1)

decoded_imgs = autoencoder.predict(input_features)

transformed= decoded_imgs.reshape(input_shape[0], input_shape[1],-1)

cv2.imwrite(dirct+"\\AE_image.jpg",transformed)