# # 模型结构的恢复
import numpy as np
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

model = load_model('./model/CNN.h5')

img = x_test[0,:,:,:]
img = np.expand_dims(img, axis = 0)
pre = model.predict(img)
print(pre)