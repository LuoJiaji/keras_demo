# 增加均衡采样
# 增加自动保存模型功能, 可以自动加载已训练模型并继续训练
# 增加日志文件的时间戳,方便管理
# 增加数据增强功能

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix
from keras import callbacks, optimizers

batch_size = 128
img_width = 128
img_height = 128
channel = 3
STATE = 'train'

STEPS = 20000
per_step_to_save = 5
train_dataset_path = './dataset/US8.1-杂质归类/'
# test_data_path = './dataset/test/'
pre_model = ''

filename = os.listdir(train_dataset_path)
n_calss = len(os.listdir(train_dataset_path))

def get_datalist(datapath):
    
    filename = os.listdir(datapath)
    datafile = []
    label = []
    cnt = 0
    print(filename)
    for i,path in enumerate(filename):
        dataname  = os.listdir(os.path.join(datapath, path))
        print(path,':',len(dataname))
        curr_path = []
        curr_label = []
        for file in dataname:
#            datafile.append(os.path.join(datapath,path,file))
            curr_path.append(datapath+path+'/'+file)
            curr_label.append(i)
        cnt += len(curr_path)
        datafile.append(curr_path)
        label.append(curr_label)
    
    print('-'*40)
    print('Data Count:',cnt)
    print('*'*40)
    
    return datafile, label


def get_random_batch(datapath, label, batchsize, n_calss, img_width, img_height, channel):
    
    data = np.zeros([batchsize, img_width, img_height, channel])
    # train_data =  train_data.astype(np.uint8)
    label_onehot = np.zeros([batchsize, n_calss])
    
    l = len(datapath)
    # i = 0

    class_cnt = []
    for i in range(l):
        class_cnt.append(len(datapath[i]))

    for i in range(batchsize):
        # image_index = random.randrange(l)
        class_index = random.randrange(l)
        # n = len(train_datapath[class_index])
        # img_index = random.randrange(n)
        img_index = random.randrange(class_cnt[class_index])
        # img = cv2.imread(train_datapath[image_index])
        # train_data[i,:,:,:]  = cv2.resize(img,(img_height,img_width))

        img = image.load_img(datapath[class_index][img_index], target_size=(img_height, img_width))
        img = image.img_to_array(img)
        # x = np.expand_dims(x, axis=0)
        # img = preprocess_input(img)
        data[i,:,:,:]  = img

        label_onehot[i, int(label[class_index][img_index])] = 1
#        print(i,image_index,train_datapath[image_index])
        # i += 1
    return data, label_onehot

# 加载文件路径
train_datapath, train_label = get_datalist(train_dataset_path)


# 数据增强
datagen = ImageDataGenerator(rotation_range = 90,  #图片随机转动的角度
                             width_shift_range = 0.1, #图片水平偏移的幅度
                             height_shift_range = 0.1, #图片竖直偏移的幅度
                             zoom_range = 0.1) #随机放大或缩小 

# quit()
# test_datalist, test_label = get_datalist(test_data_path) 

# input_shape = (img_width,img_height, 1)
# input_data = Input(shape=input_shape)
# x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_data)
# x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
# x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
# x = Flatten(name='flatten')(x)
# x = Dense(128, activation='relu', name='fc1')(x)
# x = Dense(128, activation='relu', name='fc2')(x)
# #x = Lambda(lambda x: K.dropout(x, 0.5))(x)
# x = Dense(30, activation='softmax', name='fc_output')(x)
# model = Model(input_data, x)

input_shape = (img_width, img_height, 3)
input_tensor=Input(shape=input_shape)
x = BatchNormalization(name= 'bn_1')(input_tensor)
x = Conv2D(64, (3, 3), activation='relu',padding='same', name='block1_conv1')(x)
# x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = BatchNormalization(name= 'bn_2')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = BatchNormalization(name= 'bn_3')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = BatchNormalization(name= 'bn_4')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Classification block
x = Flatten(name='flatten')(x)
# x = GlobalAveragePooling2D(name = 'Average_pooling')(x)
x = Dense(512, activation='relu', name='fc1')(x)
x = Dense(128, activation='relu', name='fc2')(x)
x = Dropout(0.8)(x)
x = Dense(n_calss, activation='softmax', name='predictions')(x)

model = Model(inputs= input_tensor, outputs = x)

model.compile(loss = 'categorical_crossentropy', optimizer = optimizers.SGD(), metrics = ['accuracy'])
model.summary()
quit()


# 判断是否需要加载模型, 并且继续训练
model_file_list = []
if os.path.exists(pre_model):
    # pre_path = 
    start = path.split('/')[-1].split('.')[0].split('_')[-1]
    start - int(start)
else:
    start = 0

print('STATE:',STATE)
if STATE == 'train':
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    t = time.strftime("%Y%m%d%H%M%S", time.localtime()) 
    log_path = './logs/' + t
    os.makedirs(log_path)

    log_file = log_path + '/logs.csv'
    acc_file = log_path + '/acc.csv'
    # log_file = './logs/logs.csv'
    # acc_file = './logs/acc.csv'
    f_log = open(log_file,'w')
    f_log.write('iter, loss, train acc' + '\n')
    f_log.flush()
    # f_acc = open(acc_file,'w')
    # f_acc.write('iter,acc'+'\n')
    # f_acc.flush()

    for it in range(start, start + STEPS):
        train_data, train_label_onehot = get_random_batch(train_datapath,
                                                        train_label,
                                                        batch_size,
                                                        n_calss,
                                                        img_width,
                                                        img_height,
                                                        channel)
        train_data  /= 225
        gen = datagen.flow(train_data, shuffle=False, batch_size=batch_size)
        train_data = next(gen)
        # train_data = train_data.astype('float')
        train_loss, train_accuracy = model.train_on_batch(train_data, train_label_onehot)
        
        # temp = model.predict(train_data)
    #    quit()

        if it % per_step_to_save == 0 or it == 0 or it + 1 == start + STEPS:
            print('iteration:',it,'loss:',train_loss,'accuracy:',train_accuracy)
            f_log.write(str(it)+','+str(train_loss)+','+str(train_accuracy)+'\n')
            f_log.flush()
        
        if it % 500 == 0:
            model_file = log_path + '/CNN_'+ str(it) + '.h5'
            model.save(model_file)
            model_file_list.append(model_file)
            
            if len(model_file_list) > 3:
                os.remove(model_file_list[0])
                del model_file_list[0]

        # print('iteration:',it,'loss:',train_loss,'accuracy:',train_accuracy)
        # f_log.write(str(it)+','+str(train_loss)+','+str(train_accuracy)+'\n')
        # f_log.flush()

        # if (it+1) % 1000 == 0 or it + 1 == STEPS:
        #     pre = []
        #     l = len(test_datalist)
        #     for i, path in enumerate(test_datalist):
        #         if i%100 == 0:
        #             print('\r','test:',i,'/',l,end = '')
                
        #         # img = cv2.imread(path)
        #         # img  = cv2.resize(img,(img_height,img_width))
        #         # # print(img.shape)
        #         # img = np.expand_dims(img,axis = 0)
        #         # print(img.shape)
        #         img = image.load_img(path, target_size=(img_width, img_width))
        #         img = image.img_to_array(img)
        #         img = np.expand_dims(img, axis=0)
        #         img = preprocess_input(img)
        #         pre += [np.argmax(model.predict(img))]
        #     pre = np.array(pre)    
        #     test_label = np.array(test_label)
        #     # print('pre shape:',pre.shape,'test label shape:',test_label.shape)
        #     acc = np.mean(pre==test_label)
        #     print('\r','*'*50)
        #     print('test accuracy:',acc)
            # f_acc.write(str(it)+','+str(acc)+'\n')
            # f_acc.flush()
    # model.save('./models/CNN2.h5')
    
    model_file = log_path + '/CNN_'+ str(it) + '.h5'
    if not os.path.exists(model_file):
        model.save(model_file)  
    f_log.close()

    # f_log.close()
    # f_acc.close()