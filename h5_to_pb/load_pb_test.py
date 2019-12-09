import tensorflow as tf
import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

 
pb_file_path = './model/CNN.pb'

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = np.expand_dims(x_train, axis = 3)
x_test = np.expand_dims(x_test, axis = 3)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


img = x_test[:2,:,:,:]

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()

    # 打开.pb模型
    with open(pb_file_path, "rb") as f:
        output_graph_def.ParseFromString(f.read())
        tensors = tf.import_graph_def(output_graph_def, name="")
        print("tensors:",tensors)

    # 在一个session中去run一个前向
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        op = sess.graph.get_operations()

        # 打印图中有的操作
        # for i,m in enumerate(op):
        #     print('op{}:'.format(i),m.values())

        input_x = sess.graph.get_tensor_by_name("input_1:0")  # 具体名称看上一段代码的input.name
        print("input_X:",input_x)

        out_softmax = sess.graph.get_tensor_by_name("fc_output/Softmax:0")  # 具体名称看上一段代码的output.name
        print("Output:",out_softmax)

        pre = sess.run(out_softmax,feed_dict={input_x: np.reshape(img,(2,28,28,1))})

        print(pre)
        # print("img_out_softmax:", img_out_softmax)
        # for i,prob in enumerate(img_out_softmax[0]):
        #     print('class {} prob:{}'.format(i,prob))
        # prediction_labels = np.argmax(img_out_softmax, axis=1)
        # print("Final class if:",prediction_labels)
        # print("prob of label:",img_out_softmax[0,prediction_labels])