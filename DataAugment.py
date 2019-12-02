
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

path = os.listdir('./img')
data = []
for p in path:
    img = cv2.imread('./img/'+ p)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = cv2.resize(img,(300,300))
    # plt.imshow(img)
    # img = np.array(img)
    data.append(img)

data = np.array(data)
# plt.show()

datagen = ImageDataGenerator(rotation_range = 90,  #图片随机转动的角度
                             width_shift_range = 0.1, #图片水平偏移的幅度
                             height_shift_range = 0.1, #图片竖直偏移的幅度
                             zoom_range = 0.2) #随机放大或缩小

gen = datagen.flow(data, batch_size=3)
x_batch = next(gen)
print(x_batch.shape)
for n in range(5):
    for i in range(3):
        x_batch = next(gen)
        plt.subplot(5, 3, n*3+i+1)
        plt.imshow(x_batch[i]/255)
plt.show()

# img = np.expand_dims(img,axis = 0)
# gen = datagen.flow(img, batch_size=3)

# for i in range(3):
#     for j in range(3):
#         x_batch = next(gen)
#         idx = (3*i) + j
#         plt.subplot(3, 3, idx+1)
#         plt.imshow(x_batch[0]/255)
# plt.show()


