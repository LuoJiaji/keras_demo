import cv2 
import numpy as np
from keras.datasets import mnist
from keras.layers import Input, BatchNormalization, add, ReLU, GlobalAveragePooling2D, Dense
from keras.layers.convolutional import Conv2D
from keras.optimizers import RMSprop, SGD
from keras import callbacks, optimizers
from keras import backend as K
from keras.models import Model,load_model
from keras.preprocessing import image
from keras.utils.vis_utils import plot_model



def RepBlock(x, filter_num, identify = True, strides = 1):

    x3 = Conv2D(filters=filter_num, kernel_size=(3, 3), strides=strides, padding='same', use_bias=False)(x)
    x3 = BatchNormalization()(x3)

    x1 = Conv2D(filters=filter_num, kernel_size=(1, 1), strides=strides, padding='same', use_bias=False)(x)
    x1 = BatchNormalization()(x1)

    if identify:
        xn = BatchNormalization()(x)
        return ReLU()(add([x3, x1, xn]))
    else:
        return ReLU()(add([x3, x1]))


num_classes = 20

input_shape = (128, 128, 1)
input_data = Input(shape=input_shape)

# stage1
x = RepBlock(input_data, filter_num=64, identify=False, strides=2)
x = RepBlock(x, filter_num=64)

# stage2
x = RepBlock(x, filter_num=128, identify=False, strides=2)
x = RepBlock(x, filter_num=128)
x = RepBlock(x, filter_num=128)
x = RepBlock(x, filter_num=128)

# stage3
x = RepBlock(x, filter_num=256, identify=False, strides=2)
x = RepBlock(x, filter_num=256)
x = RepBlock(x, filter_num=256)
x = RepBlock(x, filter_num=256)

# stage4
x = RepBlock(x, filter_num=512, identify=False, strides=2)

x = GlobalAveragePooling2D()(x)

x = Dense(num_classes, activation='softmax')(x)

model = Model(input_data, x)

model.summary()
model.save('./model/RepVGG.h5')
plot_model(model, to_file='./models_visualization/RepVGG.pdf',show_shapes=True)
