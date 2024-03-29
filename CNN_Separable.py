# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 15:20:54 2019

@author: Bllue
"""

import cv2 
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Dropout, Layer, Lambda, SeparableConv2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop, SGD, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras import callbacks, optimizers
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model,load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

input_shape = (28,28,1)
input_data = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_data)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = SeparableConv2D(32, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
x = Flatten(name='flatten')(x)
x = Dense(128, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
#x = Lambda(lambda x: K.dropout(x, 0.5))(x)
x = Dense(10, activation='softmax', name='fc_output')(x)
model = Model(input_data, x)
quit()
#model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(), metrics = ['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer = RMSprop(), metrics = ['accuracy'])
model.summary()
quit()
model.fit(x_train, y_train, epochs=40, batch_size=128)

pre = model.predict(x_test)
pre = np.argmax(pre, axis = 1)
y_test = np.argmax(y_test, axis = 1)
acc = np.mean(pre == y_test)
print('accuracy:', acc)

# 98.11%
# 97.48%
# 98.18%
