import cv2 
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Flatten, Dense, Dropout, Layer,Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import RMSprop, SGD
from keras import callbacks, optimizers
from keras.utils import np_utils
from keras import backend as K
from keras.models import Model,load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train *= 10
x_test *= 10
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

input_shape = (28,28,1)
input_data = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_data)
x = BatchNormalization(name= 'bn_block1_conv1')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = BatchNormalization(name= 'bn_block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

x = Flatten(name='flatten')(x)
x = Dense(128, activation='relu', name='fc1')(x)
#x = Dropout(0.5)(x)
x = Dense(128, activation='relu', name='fc2')(x)
#x = Lambda(lambda x: K.dropout(x, 0.5))(x)
x = Dropout(0.5)(x)
x = Dense(10, activation='softmax', name='fc_output')(x)
model = Model(input_data, x)

model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(), metrics = ['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=40, batch_size=128)

pre = model.predict(x_test)

pre = np.argmax(pre, axis = 1)
label = np.argmax(y_test, axis = 1)
acc =np.mean(pre==label)


#img = x_test[0,:,:,:]
#img = np.expand_dims(img, axis=0)

#model_inter = Model(inputs=model.input, outputs=model.get_layer('block1_conv1').output)
#interlayer_output = model_inter.predict(img)
#block1_conv1 = interlayer_output[0,:,:,:]

#model_inter = Model(inputs=model.input, outputs=model.get_layer('bn_block1_conv1').output)
#interlayer_output = model_inter.predict(img)
#bn_block1_conv1 = interlayer_output[0,:,:,:]

#model_inter = Model(inputs=model.input, outputs=model.get_layer('block1_conv2').output)
#interlayer_output = model_inter.predict(img)
#block1_conv2 = interlayer_output[0,:,:,:]

#model_inter = Model(inputs=model.input, outputs=model.get_layer('bn_block1_conv2').output)
#interlayer_output = model_inter.predict(img)
#bn_block1_conv2 = interlayer_output[0,:,:,:]

