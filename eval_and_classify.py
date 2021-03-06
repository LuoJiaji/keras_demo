import os
import numpy as np 
import time
from shutil import copyfile
from keras.models import Model, load_model
from keras.preprocessing import image

img_width = 100
img_height = 100
model_path = './models/CNN.h5'
train_dataset_path = 'dataset'
testdata_path = './test_data/'
t = time.strftime("%Y%m%d-%H%M%S", time.localtime())
t = t[2:] 
testresult_path = './result/test_result_of_' + testdata_path.split('/')[1] + '_'+ t+ '/'
dir_list = os.listdir(train_dataset_path)

if not os.path.exists(testresult_path):
    os.makedirs(testresult_path)

data_path = os.listdir(testdata_path)
l = len(data_path)

print('test dataset:', testdata_path)
print('category:', len(dir_list), dir_list)
print('number of test data:', l)

# data_path = [ testdata_path+p for p in data_path]

# model = load_model(model_path)

for i, p in enumerate(data_path):
    img = image.load_img(testdata_path + p, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img /= 255
    # print(img.shape)
    pre = model.predict(img)
    # pre = [0.2, 0.3]
    result = np.argmax(pre)
    # result = 1
    # time.sleep(2)
    if not os.path.exists(testresult_path + dir_list[result]):
        os.makedirs(testresult_path +  dir_list[result])
    copyfile(testdata_path + p, testresult_path +  dir_list[result] + '/' + p)

    if not os.path.exists(testresult_path + 'result_in_csv'):
        os.makedirs(testresult_path + 'result_in_csv')
    
    with open(testresult_path + 'result_in_csv/' + dir_list[result] + '.csv','a+') as f:
        f.write(p +',' + str(np.max(pre)) + ','  + '\n')

        
    if i % 100 == 0:
        print('\r', 'test:', i, '/', l-1, end = '')


    



