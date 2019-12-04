import os
import numpy as np 
from shutil import copyfile
from keras.models import Model, load_model
from keras.preprocessing import image

img_width = 100
img_height = 100
model_path = './logs/********/CNN.h5'
train_dataset_path = 'dataset'
testdata_path = './test_data/'
testresult_path = './rest_result/'
dir_list = os.listdir(train_dataset_path)
print('dir list:', dir_list)

if not os.path.exists(testresult_path):
    os.makedirs(testresult_path)

data_path = os.listdir(testdata_path)
print('number of test data:', len(data_path))
# data_path = [ testdata_path+p for p in data_path]

model = load_model(model_path)

for p in data_path:
    img = image.load_img(testdata_path + p, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    print(img.shape)
    pre = model.predict(img)
    result = np.argmax(pre)
    # result = 1
    if not os.path.exists(testresult_path + dir_list[result]):
        os.makedirs(testresult_path +  dir_list[result])
    copyfile(testdata_path + p, testresult_path +  dir_list[result] + '/' + p)

    



