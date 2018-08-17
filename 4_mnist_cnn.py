import numpy as np
import tensorflow as tf
import tensorlayer as tl
import time

#[-1,28,28,1]中的第一个值为图片的数量,-1是让python自己推算应该有多少张图片
x_train,y_train,x_val,y_val,x_test,y_test = tl.files.load_mnist_dataset(shape=[-1,28,28,1])

sess = tf.InteractiveSession()

#设置超参数                                                                                                                      
batch_size = 128
learning_rate = 0.001
n_epoch = 200
print_freq = 5

x = tf.placeholder(tf.float32,shape = [batch_size,28,28,1])
y_ = tf.placeholder(tf.int64,shape = [batch_size,])

#定义网络模型
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

#训练模型
for epoch in range(n_epoch):
    start_time = time.time()
    
    #训练一个epoch
    for x_train_a,y_train_a in tl.iterate.minibatches(x_train, y_train, batch_size, shuffle = True):
        feed_dict = {x:x_train_a,y_:y_train_a}
        #启用Dropout,使用keep值作为DropoutLayer的概率
        feed_dict.update(network.all_drop)
        sess.run(train_op, feed_dict = feed_dict)
         
    if epoch + 1 == 1 or (epoch+1)%print_freq == 0:
        #每print_freq个,打印时间
        print('Epoch %d of %d took %fs'%(epoch+1,n_epoch,time.time()-start_time))
        
        #用训练集做测试
        train_loss,train_acc,n_batch = 0,0,0
        for x_train_a,y_train_a in tl.iterate.minibatches(x_train, y_train, batch_size, shuffle = False):
            
            #关闭Dropout,把Dropout中的keep设为1
            dp_dict = tl.utils.dict_to_one(network.all_drop)
            
            feed_dict = {x:x_train_a,y_:y_train_a}
            #启用Dropout,使用keep值作为DropoutLayer的概率
            feed_dict.update(dp_dict)
            err,ac = sess.run([cost,acc], feed_dict = feed_dict)
            train_loss +=err;train_acc +=ac;n_batch+=1
        print(' train_loss: %f'%(train_loss/n_batch))
        print(' train_acc: %f'%(train_acc/n_batch))
        
        
        #用验证集做测试
        val_loss,val_acc,n_batch = 0,0,0
        for x_val_a,y_val_a in tl.iterate.minibatches(x_val,y_val,batch_size,shuffle = False):
            #关闭Dropout,把Dropout中的keep设为1
            dp_dict = tl.utils.dict_to_one(network.all_drop)
            
            feed_dict = {x:x_val_a,y_:y_val_a}
            #启用Dropout,使用keep值作为DropoutLayer的概率
            feed_dict.update(dp_dict)
            err,ac = sess.run([cost,acc], feed_dict = feed_dict)
            val_loss +=err;val_acc +=ac;n_batch+=1
        print(' val_loss: %f'%(val_loss/n_batch))
        print(' val_acc: %f'%(val_acc/n_batch))
      
#用测试集做测试
test_loss,test_acc,n_batch = 0,0,0
for x_test_a,y_test_a in tl.iterate.minibatches(x_test,y_test,batch_size,shuffle = True):
    #关闭Dropout,把Dropout中的keep设为1
    dp_dict = tl.utils.dict_to_one(network.all_drop)
    
    feed_dict = {x:x_val_a,y_:y_val_a}
    #启用Dropout,使用keep值作为DropoutLayer的概率
    feed_dict.update(dp_dict)
    err,ac = sess.run([cost,acc], feed_dict = feed_dict)
    test_loss +=err;test_acc +=ac;n_batch+=1
print(' test_loss: %f'%(test_loss/n_batch))
print(' test_acc: %f'%(test_acc/n_batch))
        

'''保存模型'''
tl.files.save_npz(network.all_params,name = 'model.npz')
sess.close()




