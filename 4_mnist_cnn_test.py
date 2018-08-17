import os
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorlayer as tl

def loaddata(filename):
    #转换成灰度图也就是单通道。
    img=Image.open(filename).resize((28,28)).convert('L')
    #取出数据
    array=np.asarray(img,dtype="float32")
    #取反
    array=abs(255-array) #如果图像和MNIST里的一样，这里要注释 
    array = np.reshape(array,[1,28,28,1])
    return array


def model(x,y_,sess):

    learning_rate = 0.001
    network = tl.layers.InputLayer(x,name = 'input')
    network = tl.layers.Conv2d(
                network,n_filter=32,
                filter_size=(5, 5),
                strides=(1, 1),
                act=tf.nn.relu,
                padding='SAME',
                name = 'cnn1')
    network = tl.layers.MaxPool2d(
                network, 
                filter_size=(2, 2), 
                strides=(2, 2), 
                padding='SAME', 
                name='pool1')
    network = tl.layers.Conv2d(
                network,
                n_filter=64,
                filter_size=(5, 5),
                strides=(1, 1),
                act=tf.nn.relu,
                padding='SAME',
                name = 'cnn2')
    network = tl.layers.MaxPool2d(
                network, 
                filter_size=(2, 2), 
                strides=(2, 2), 
                padding='SAME', 
                name='pool2')

    network = tl.layers.FlattenLayer(network,name = 'FlattenLayer')  #把张量拉成一个向量
    network = tl.layers.DropoutLayer(network, keep = 0.5, name = 'drop1')
    network = tl.layers.DenseLayer(network, 256, tf.nn.relu,name = 'relu1')
    network = tl.layers.DropoutLayer(network, keep = 0.5, name = 'drop2')
    network = tl.layers.DenseLayer(network, 10, tf.identity,name = 'outputs')

    y = network.outputs

    #定义损失函数
    cost = tl.cost.cross_entropy(y,y_,name = 'cost')
    correct_prediction = tf.equal(tf.argmax(y,1),y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #定义优化器
    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost,var_list=train_params)

    #初始化参数
    tl.layers.initialize_global_variables(sess)

    return network

def main():
    
    sess = tf.InteractiveSession()
    x = tf.placeholder(tf.float32,shape = [None,28,28,1])
    y_ = tf.placeholder(tf.int64,shape = [None,])
    
    #构建模型骨架
    network = model(x,y_,sess)

    y = network.outputs
    
    y_op = tf.argmax(tf.nn.softmax(y),1)  #对网络结构的输出y做softmax处理,然后用tf.argmax()取出概率最大的一个

    #加载模型参数
    tl.files.load_and_assign_npz(sess, "model.npz", network)

    #测试一张图片
    #predict_data = loaddata('img/9.jpg') 
    #print('预测结果:',tl.utils.predict(sess,network,predict_data,x,y_op)[0])

    #读入数据
    filelist = os.listdir('img')
    filelist.sort(key=lambda x: int(x[:-4]))
    print('文件中的图片有:',filelist)
    for file in filelist:
        predict_data = loaddata(os.path.join('img',file))    
        
        #进行预测
        print('预测结果:',tl.utils.predict(sess,network,predict_data,x,y_op)[0])


if __name__ == '__main__':
    main()