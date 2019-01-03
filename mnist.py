# -x- coding: utf-8 -x-
"""
Spyder Editor

This is a temporary script file.
"""
#from tensorflow.examples.tutorials.mnist import input_data
#mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)
#
#import numpy as np
#import tensorflow as tf
#
#sess=tf.InteractiveSession()
#
##单隐含层神经网络实现，准确率在98%
#in_units=784
#h1_units=300
#W1=tf.Variable(tf.truncated_normal([in_units,h1_units],stddev=0.1))
#b1=tf.Variable(tf.zeros(h1_units))
#W2=tf.Variable(tf.zeros([h1_units,10]))
#b2=tf.Variable(tf.zeros([10]))
#
#x=tf.placeholder(tf.float32,[None,in_units])
#keep_prop=tf.placeholder(tf.float32)
#
#hidden1=tf.nn.relu(tf.matmul(x,W1)+b1)
#hidden1_drop=tf.nn.dropout(hidden1,keep_prop)
#y=tf.nn.softmax(tf.matmul(hidden1_drop,W2)+b2)
#
#y_=tf.placeholder(tf.float32,[None,10])#存放的是标签结果
#cross_entroy=tf.reduce_mean(-tf.reduce_sum(y_xtf.log(y),reduction_indices=[1]))
#train_step=tf.train.AdagradOptimizer(0.3).minimize(cross_entroy)
#
#tf.global_variables_initializer().run()
#for i in range(3000):
#    batch_xs,batch_ys=mnist.train.next_batch(128)
#    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys,keep_prop:0.75})
#    loss=sess.run(cross_entroy,feed_dict={x:mnist.train.images,y_:mnist.train.labels,
#                                          keep_prop:1.0})
#    print(str(i),"=",str(loss))
#    
#correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prop:1.0}))
#
#
##卷积神经网络，两个隐含层加一个全连接层 训练好了准确率可达99%
#def weight_variable(shape):
#    initial=tf.truncated_normal(shape,stddev=0.1)
#    return tf.Variable(initial)
#
#def bias_variable(shape):
#    initial=tf.constant(0.1,shape=shape)
#    return tf.Variable(initial)
#
#def conv2d(x,W):
#    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
#
#def max_pool_2x2(x):
#    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
#
#x=tf.placeholder(tf.float32,[None,784])
#y_=tf.placeholder(tf.float32,[None,10])
#x_image=tf.reshape(x,[-1,28,28,1])
#
#W_conv1=weight_variable([5,5,1,32])
#b_conv1=bias_variable([32])
#h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
#h_pool1=max_pool_2x2(h_conv1)
#
#W_conv2=weight_variable([5,5,32,64])
#b_conv2=bias_variable([64])
#h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
#h_pool2=max_pool_2x2(h_conv2)
#
#W_fc1=weight_variable([7x7x64,1024])
#b_fc1=bias_variable([1024])
#h_pool2_flat=tf.reshape(h_pool2,[-1,7x7x64])
#h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
#
#keep_prop=tf.placeholder(tf.float32)
#h_fc1_drop=tf.nn.dropout(h_fc1,keep_prop)
#
#W_fc2=weight_variable([1024,10])
#b_fc2=bias_variable([10])
#y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
#
#cross_entropy=tf.reduce_mean(-tf.reduce_sum(y_xtf.log(y_conv),reduction_indices=[1]))
#
#train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#
#correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#
#tf.global_variables_initializer().run()
#for i in range(20000):
#    batch=mnist.train.next_batch(50)
#    if i%100==0:
#        train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prop:1.0})
#        print("step %d, train_accuracy %g"%(i,train_accuracy) )   
#    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prop:0.5})
#
#test_accuracy=accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prop:1.0})
#print("step %d, train_accuracy %g"%test_accuracy )
    






#Alexnet
#from datetime import datetime
#import math
#import time
#import tensorflow as tf
#
#batch_size=32
#num_batches=2
#
#def print_activations(t):
#    print(t.op.name,' ',t.get_shape().as_list())
#
#
#def interence(images):
#    parameters=[]
#    
#    #第1个卷积层
#    with tf.name_scope('conv1') as scope:
#        kernel=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=tf.float32,
#                                               stddev=1e-1),name='weights')
#        conv=tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
#        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),
#                           trainable=True,name='biases')
#        bias=tf.nn.bias_add(conv,biases)
#        conv1=tf.nn.relu(bias,name=scope)
#        print_activations(conv1)
#        parameters+=[kernel,biases]
#
#    lrn1=tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')
#    pool1=tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],
#                         padding='VALID',name='pool1')
#    print_activations(pool1)
#        
#    
#    #第2个卷积层    
#    with tf.name_scope('conv2') as scope:
#        kernel=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,
#                                               stddev=1e-1),name='weights')
#        conv=tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
#        biases=tf.Variable(tf.constant(0.0,shape=[192],dtype=tf.float32),
#                                       trainable=True,name='biases')
#        bias=tf.nn.bias_add(conv,biases)
#        conv2=tf.nn.relu(bias,name=scope)
#        print_activations(conv2)
#        parameters+=[kernel,biases]
#        
#    lrn2=tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')
#    pool2=tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],
#                         padding='VALID',name='pool2')
#    print_activations(pool2)
#    
#        
#    #第3个卷积层
#    with tf.name_scope('conv3') as scope:
#        kernel=tf.Variable(tf.truncated_normal([3,3,192,384],dtype=tf.float32,
#                                               stddev=1e-1),name='weights')
#        conv=tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
#        biases=tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),
#                           trainable=True,name='biases')
#        bias=tf.nn.bias_add(conv,biases)
#        conv3=tf.nn.relu(bias,name=scope)
#        parameters+=[kernel,biases]
#        print_activations(conv3)
#
#    
#    #第4个卷积层    
#    with tf.name_scope('conv4') as scope:
#        kernel=tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,
#                                               stddev=1e-1),name='weights')
#        conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
#        biases=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),
#                           trainable=True,name='biases')
#        bias=tf.nn.bias_add(conv,biases)
#        conv4=tf.nn.relu(bias,name=scope)
#        parameters+=[kernel,biases]
#        print_activations(conv4)
#
#        
#    #第5个卷积层    
#    with tf.name_scope('conv5') as scope:
#        kernel=tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,
#                                               stddev=1e-1),name='weights')
#        conv=tf.nn.conv2d(conv4,kernel,[1,1,1,1],padding='SAME')
#        biases=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),
#                           trainable=True,name='biases')
#        bias=tf.nn.bias_add(conv,biases)
#        conv5=tf.nn.relu(bias,name=scope)
#        parameters+=[kernel,biases]
#        print_activations(conv5)
#    
#        
#    pool5=tf.nn.max_pool(conv5,[1,3,3,1],strides=[1,2,2,1],padding='VALID',
#                         name='pool5')
#    print_activations(pool5)
#        
#    return pool5,parameters
#
#def time_tensorflow_run(session,target,info_string):
#    num_steps_burn_in=1
#    total_duration=0.0
#    total_duration_squared=0.0
#    for i in range(num_batches+num_steps_burn_in):
#        start_time=time.time()
#        _=session.run(target)
#        duration=time.time()-start_time
#        if i>num_steps_burn_in:
#            if not i%10:
#                print('%s:step %d,duration=%.3f' % (datetime.now(),
#                                                    i-num_steps_burn_in,duration))
#                
#            total_duration+=duration
#            total_duration_squared+=durationxduration
#            
#    mn=total_duration/num_batches
#    print(mn)
#    vr=total_duration_squared/num_batches-mnxmn
#    print(vr)
#    sd=math.sqrt(vr)
#    print('%s:%s across %d steps,%.3f+/-%.3f sec /batch' % (datetime.now(),
#                                                            info_string,
#                                                            num_batches,
#                                                            mn,sd))
#    
#
#def run_benchmark():
#    with tf.Graph().as_default():
#        image_size=224
#        images=tf.Variable(tf.random_normal([batch_size,
#                                             image_size,
#                                             image_size,
#                                             3],dtype=tf.float32,stddev=1e-1))
#        pool5,parameters=interence(images)
#        
#        init=tf.global_variables_initializer()
#        sess=tf.Session()
#        sess.run(init)
#        time_tensorflow_run(sess,pool5,'Forward')
#        
#        objective=tf.nn.l2_loss(pool5)
#        grad=tf.gradients(objective,parameters)
#        time_tensorflow_run(sess,grad,'Forward_backward')
#        
#run_benchmark()
    
    
       
#VGG-16
#from datetime import datetime
#import math
#import time
#import tensorflow as tf
#
##卷积层
#def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
#    n_in=input_op.get_shape()[-1].value
#    with tf.name_scope(name) as scope:
#        kernel=tf.get_variable(scope+'w',shape=[kh,kw,n_in,n_out],
#                               dtype=tf.float32,
#                               initializer=tf.contrib.layers.xavier_initializer_conv2d())
#        conv=tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
#        bias_init_val=tf.constant(0.0,shape=[n_out],dtype=tf.float32)
#        biases=tf.Variable(bias_init_val,trainable=True,name='b')
#        z=tf.nn.bias_add(conv,biases)
#        activation=tf.nn.relu(z,name=scope)
#        p+=[kernel,biases]
#        return activation
#
##全连接层
#def fc_op(input_op,name,n_out,p):
#    n_in=input_op.get_shape()[-1].value
#    
#    with tf.name_scope(name) as scope:
#        kernel=tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,
#                               initializer=tf.contrib.layers.xavier_initializer())
#        biases=tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),
#                           name='b')
#        activation=tf.nn.relu_layer(input_op,kernel,biases,name=scope)
#        p+=[kernel,biases]
#        return activation
#   
##最大池化层    
#def mpool_op(input_op,name,kh,kw,dh,dw):
#    return tf.nn.max_pool(input_op,
#                          ksize=[1,kh,kw,1],
#                          strides=[1,dh,dw,1],
#                          padding='SAME',
#                          name=name)
#    
#def inference_op(input_op,keep_prop):
#    p=[]
#    #第1段网络，两个卷积层和一个最大池化层
#    conv1_1=conv_op(input_op,name='conv1_1',kh=3,kw=3,n_out=64,
#                    dh=1,dw=1,p=p)
#    conv1_2=conv_op(conv1_1,name='conv1_2',kh=3,kw=3,n_out=64,
#                    dh=1,dw=1,p=p)
#    pool1=mpool_op(conv1_2,name='pool1',kh=2,kw=2,dw=2,dh=2)
#     #第2段网络，两个卷积层和一个最大池化层
#    conv2_1=conv_op(pool1,name='conv2_1',kh=3,kw=3,n_out=128,
#                    dh=1,dw=1,p=p)
#    conv2_2=conv_op(conv2_1,name='conv2_2',kh=3,kw=3,n_out=128,
#                    dh=1,dw=1,p=p)
#    pool2=mpool_op(conv2_2,name='pool2',kh=2,kw=2,dw=2,dh=2)
#     #第3段网络，三个卷积层和一个最大池化层
#    conv3_1=conv_op(pool2,name='conv3_1',kh=3,kw=3,n_out=256,
#                    dh=1,dw=1,p=p)
#    conv3_2=conv_op(conv3_1,name='conv3_2',kh=3,kw=3,n_out=256,
#                    dh=1,dw=1,p=p)
#    conv3_3=conv_op(conv3_2,name='conv3_3',kh=3,kw=3,n_out=256,
#                    dh=1,dw=1,p=p)
#    pool3=mpool_op(conv3_3,name='pool3',kh=2,kw=2,dw=2,dh=2)    
#    #第4段网络，三个卷积层和一个最大池化层
#    conv4_1=conv_op(pool3,name='conv4_1',kh=3,kw=3,n_out=512,
#                    dh=1,dw=1,p=p)
#    conv4_2=conv_op(conv4_1,name='conv4_2',kh=3,kw=3,n_out=512,
#                    dh=1,dw=1,p=p)
#    conv4_3=conv_op(conv4_2,name='conv4_3',kh=3,kw=3,n_out=512,
#                    dh=1,dw=1,p=p)
#    pool4=mpool_op(conv4_3,name='pool4',kh=2,kw=2,dw=2,dh=2)  
#    #第5段网络，三个卷积层和一个最大池化层
#    conv5_1=conv_op(pool4,name='conv5_1',kh=3,kw=3,n_out=512,
#                    dh=1,dw=1,p=p)
#    conv5_2=conv_op(conv5_1,name='conv5_2',kh=3,kw=3,n_out=512,
#                    dh=1,dw=1,p=p)
#    conv5_3=conv_op(conv5_2,name='conv5_3',kh=3,kw=3,n_out=512,
#                    dh=1,dw=1,p=p)
#    pool5=mpool_op(conv5_3,name='pool5',kh=2,kw=2,dw=2,dh=2)
#    
#    shp=pool5.get_shape()
#    flattened_shape=shp[1].valuexshp[2].valuexshp[3].value
#    resh1=tf.reshape(pool5,[-1,flattened_shape],name='resh1')
#    
#    #全连接层
#    fc6=fc_op(resh1,name='fc6',n_out=4096,p=p)
#    fc6_drop=tf.nn.dropout(fc6,keep_prob=keep_prop,name='fc6_drop')
#    
#    fc7=fc_op(fc6_drop,name='fc7',n_out=4096,p=p)
#    fc7_drop=tf.nn.dropout(fc7,keep_prob=keep_prop,name='fc7_drop')
#    
#    #softmax
#    fc8=fc_op(fc7_drop,name='fc8',n_out=1000,p=p)
#    softmax=tf.nn.softmax(fc8)
#    predictions=tf.arg_max(softmax,1)
#    
#    return predictions,softmax,fc8,p
#
#def time_tensorflow_run(session,target,feed,infor_string):
#    num_steps_burn_in=0
#    total_duration=0.0
#    total_duration_squared=0.0
#    for i in range(num_batches+num_steps_burn_in):
#        start_time=time.time()
#        _=session.run(target,feed_dict=feed)
#        duration=time.time()-start_time
#        if i>=num_steps_burn_in:
#            if not i%10:
#                print('%s:step %d,duration=%.3f' % (datetime.now(),
#                                                    i-num_steps_burn_in,duration))
#                
#            total_duration+=duration
#            total_duration_squared+=durationxduration
#            
#    mn=total_duration/num_batches
#    print(mn)
#    vr=total_duration_squared/num_batches-mnxmn
#    print(vr)
#    sd=math.sqrt(vr)
#    print('%s:%s across %d steps,%.3f+/-%.3f sec /batch' % (datetime.now(),
#                                                            infor_string,
#                                                            num_batches,
#                                                            mn,sd))
#    
#def run_benchmark():
#    with tf.Graph().as_default():
#        image_size=224
#        images=tf.Variable(tf.random_normal([batch_size,
#                                             image_size,
#                                             image_size,
#                                             3],dtype=tf.float32,stddev=1e-1))
#        keep_prop=tf.placeholder(tf.float32)
#        predictions,softmax,fc8,p=inference_op(images,keep_prop)     
#        init=tf.global_variables_initializer()
#        sess=tf.Session()
#        sess.run(init)
#        
#        time_tensorflow_run(sess,predictions,{keep_prop:1.0},'Forward')
#        
#        objective=tf.nn.l2_loss(fc8)
#        grad=tf.gradients(objective,p)
#        time_tensorflow_run(sess,grad,{keep_prop:0.5},'Forward_backward')
# 
#
#batch_size=32
#num_batches=2       
#run_benchmark()


#Google InceptionV3
import tensorflow as tf
slim=tf.contrib.slim
trunc_normal=lambda stddev:tf.truncated_normal_initializer(0.0,stddev)

#def inveption_v3_arg_scope(weight_decay=0.00004,
#                           stddev=0.1,
#                           batch_norm_var_collection='moving_vars'):
#    batch_norm_params={
#            'decay':0.9997,
#            'epsilon':0.001,
#            'updates_collections':tf.GraphKeys.UPDATE_OPS,
#            'variables_collections':{
#                    'beta':None,
#                    'gamma':None,
#                    'moving_mean':[batch_norm_var_collection],
#                    'moving_variance':[batch_norm_var_collection],
#                    }
#            }
#            
#    with slim.arg_scope([slim.conv2d,slim.fully_connected],
#                        weights_regularizer=slim.l2_regularizer(weight_decay)):
#        with slim.arg_scope([slim.conv2d],                        
#                        weights_initializer=tf.truncated_normal_initializer(
#                                stddev=stddev),
#                        activation_fn=tf.nn.relu,
#                        normalizer_fn=slim.batch_norm,
#                        normalizer_params=batch_norm_params) as sc:
#return sc
    
def inception_v3_base(inputs,scope=None):
    end_points={}
    with tf.variable_scope(scope,'InceptionV3',[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                            stride=1,padding='VALID'):
            net=slim.conv2d(inputs,32,[3,3],stride=2,scope='Conv2d_1a_3x3')
            print(net.name)
            net=slim.conv2d(net,32,[3,3],scope='Conv2d_2a_3x3')
            net=slim.conv2d(net,64,[3,3],padding='SAME',scope='Conv2d_2b_3x3')
            net=slim.max_pool2d(net,[3,3],stride=2,scope='MaxPool_3a_3x3')
            net=slim.conv2d(net,80,[1,1],scope='Conv2d_3b_1x1')
            net=slim.conv2d(net,192,[3,3],scope='Conv2d_4a_3x3')
            net=slim.max_pool2d(net,[3,3],stride=2,scope='MaxPool_5a_3x3')
            
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                            stride=1,padding='SAME'):
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='AvfPool_0a_3x3')
                    branch_3=slim.conv2d(branch_3,32,[1,1],scope='Conv2d_0b_1x1')
                    
            net=tf.concat([branch_0,branch_1,branch_2,branch_3])
            
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='AvfPool_0a_3x3')
                    branch_3=slim.conv2d(branch_3,64,[1,1],scope='Conv2d_0b_1x1')
                    
            net=tf.concat([branch_0,branch_1,branch_2,branch_3])
            
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1x1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5x5')
                with tf.variable_scope('Branch_2'):
                    branch_2=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3x3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='AvfPool_0a_3x3')
                    branch_3=slim.conv2d(branch_3,64,[1,1],scope='Conv2d_0b_1x1')
                    
                net=tf.concat([branch_0,branch_1,branch_2,branch_3])
        
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,384,[3,3],stride=2,
                                         padding='VALID',scope='Conv2d_0a_3x3')
                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1x1')
                    branch_1=slim.conv2d(branch_1,96,[3,3],scope='Conv2d_0b_3x3')
                    branch_1=slim.conv2d(branch_1,96,[3,3],stride=2,
                                         padding='VALID',scope='Conv2d_0c_3x3')
                with tf.variable_scope('Branch_2'):
                    branch_2=slim.max_pool2d(net,[3,3],stride=2,
                                             padding='VALID',scope='MaxPool_0a_3x3')
                    
                net=tf.concat([branch_0,branch_1,branch_2])
                
            end_points['Mixed_6e']=net
            

image_size=224
batch_size=24
images=tf.Variable(tf.random_normal([batch_size,
                                     image_size,
                                     image_size,
                                     3],dtype=tf.float32,stddev=1e-1))

inception_v3_base(images,scope='my')





import tensorflow as tf

with tf.name_scope('graph') as scope:
     matrix1 = tf.constant([[3., 3.]],name ='matrix1')  #1 row by 2 column
     matrix2 = tf.constant([[2.],[2.]],name ='matrix2') # 2 row by 1 column
     product = tf.matmul(matrix1, matrix2,name='product')
  
sess = tf.Session()

writer = tf.summary.FileWriter("d:logs/", sess.graph)

init = tf.global_variables_initializer()

sess.run(init)

import tensorflow as tf
import numpy as np

## prepare the original data
with tf.name_scope('data'):
     x_data = np.random.rand(100).astype(np.float32)
     y_data = 0.3*x_data+0.1
##creat parameters
with tf.name_scope('parameters'):
     weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
     bias = tf.Variable(tf.zeros([1]))
##get y_prediction
with tf.name_scope('y_prediction'):
     y_prediction = weight*x_data+bias
##compute the loss
with tf.name_scope('loss'):
     loss = tf.reduce_mean(tf.square(y_data-y_prediction))
##creat optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
#creat train ,minimize the loss 
with tf.name_scope('train'):
     train = optimizer.minimize(loss)
#creat init
with tf.name_scope('init'): 
     init = tf.global_variables_initializer()
##creat a Session 
sess = tf.Session()
##initialize
writer = tf.summary.FileWriter("d:/logs/", sess.graph)
sess.run(init)
## Loop
for step  in  range(101):
    sess.run(train)
#    if step %10==0:
#        print step,'weight:',sess.run(weight),'bias:',sess.run(bias)

    


import tensorflow as tf
import numpy as np

## prepare the original data
with tf.name_scope('data'):
     x_data = np.random.rand(100).astype(np.float32)
     y_data = 0.3*x_data+0.1
##creat parameters
with tf.name_scope('parameters'):
     with tf.name_scope('weights'):
            weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
            tf.summary.histogram('weight',weight)
     with tf.name_scope('biases'):
           bias = tf.Variable(tf.zeros([1]))
           tf.summary.histogram('bias',bias)
##get y_prediction
with tf.name_scope('y_prediction'):
     y_prediction = weight*x_data+bias
##compute the loss
with tf.name_scope('loss'):
     loss = tf.reduce_mean(tf.square(y_data-y_prediction))
     tf.summary.scalar('loss',loss)
##creat optimizer
optimizer = tf.train.GradientDescentOptimizer(0.5)
#creat train ,minimize the loss 
with tf.name_scope('train'):
     train = optimizer.minimize(loss)
#creat init
with tf.name_scope('init'): 
     init = tf.global_variables_initializer()
##creat a Session 
sess = tf.Session()
#merged
merged = tf.summary.merge_all()
##initialize
writer = tf.summary.FileWriter('d:/logs/', sess.graph)
sess.run(init)
## Loop
for step  in  range(101):
    sess.run(train)
    rs=sess.run(merged)
    writer.add_summary(rs, step)
    
    

import tensorflow as tf

with tf.name_scope('input1'):
    input1=tf.constant([1.0,2.0,3.0],name='input1')

with tf.name_scope('input2'):
    input2=tf.Variable(tf.random_uniform([3]),name='input2')
output=tf.add_n([input1,input2],name='add')

writer=tf.summary.FileWriter('d:/log',tf.get_default_graph())
writer.close();

with tf.variable_scope('foo'):
    a=tf.get_variable('bar',[1])
    print(a.name)


with tf.variable_scope('bar'):
    b=tf.get_variable('bar',[2])
    print(b.name)
    
with tf.name_scope('a'):
    a=tf.Variable([3])
    print(a.name)
    
    a=tf.get_variable('b',[1])
    print(a.name)
    
    
with tf.name_scope('b'):
    tf.get_variable('b',[1])
    
    
    
    
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference

INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500

def train(mnist):
    with tf.name_scope('input1'):
        x=tf.placeholder(tf.float32,[None,mnist_inference.INPUT_NODE],
                         name='x-input')
        y_=tf.placeholder(tf.float32,[None,mnist_inference.OUTPUT_NODE],
                          name='y-input')
        
    regularizer=tf.contrib.layers.l2_regularize(REGULARAZTION_RATE)# 正则项
    y=mnist_inference.inference(x,regularizer) #前向传播
    global_step=tf.Variable(0,trainable=False) #用来记录梯度下降了多少次
    
    with tf.name_scope('moving_average'):
        variable_average=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,
                                                           global_step)
        variable_average_op=variable_average.apply(tf.trainable_variables())#对训练后的参数进行指数加权平均
        
    with tf.name_scope('loss_function'):
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
        cross_entropy_mean=tf.reduce_mean(cross_entropy)
        loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses')) #代价函数，后面这部分加上了正则项
        
    with tf.name_scope('train_step'):
        learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,
                                                 global_step,
                                                 mnist.train.num_examples/BATCH_SIZE,
                                                 LEARNING_RATE_DECAY,
                                                 staircase=True) #学习率根据训练的次数进行指数衰减
        train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                                                    global_step=global_step) #训练的时候会对训练次数进行更新
        with tf.control_dependencies([train_step,variable_average_op]):
            train_op=tf.no_op(name='train')   #将训练和参数指数平均放到了一起，并且创建了一个空Op，这样调用空Op时会对训练和指数平均的操作进行调用
            
                                    
            
            
    writer=tf.train.summaryWriter('d:/log',tf.get_default_graph())
    writer.close()
    

import tensorflow as tf

g=tf.Graph()
with g.as_default():
    a=tf.constant(1)
    g.add_to_collection('tt',a)
    print(g.collections)


x=tf.constant([[0.7,0.9]])
#tf.assign(x,[1.0,2.0])
ini=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(ini)
    print(sess.run(x))
    
    
    
import tensorflow as tf

from numpy.random import RandomState

batch_size=8

W1=tf.Variable(tf.random_normal([2,3],stddev=1.0,seed=1))
W2=tf.Variable(tf.random_normal([3,1],stddev=1.0,seed=1))

x=tf.placeholder(tf.float32,shape=(None,2),name='x_input')
y_=tf.placeholder(tf.float32,shape=(None,1),name='y_input')

a=tf.matmul(x,W1)
y=tf.matmul(a,W2)

cross_entropy=-tf.reduce_mean(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))

train_step=tf.train.AdamOptimizer(0.001).minimize(cross_entropy)




import tensorflow as tf

def get_weight(shape,lambd):
    var=tf.Variable(tf.truncated_normal(shape,tf.float32))
    tf.add_to_collection('losses',tf.contrib.l2_regularizer(lambd)(var))
    return var

x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

batch_size=2

layer_dimension=[2,10,10,10,1]

n_layers=len(layer_dimension)

cur_layers=x

in_dimension=layer_dimension[0]

for i in range(1,n_layers):
    out_dimension=layer_dimension[i]
    weight=get_weight([in_dimension,out_dimension],0.001)
    bias=tf.Variable(tf.constant(0.1,shape=[out_dimension]))
    cur_layers=tf.nn.relu(tf.matmul(cur_layers,weight)+bias)
    in_dimension=layer_dimension[i]
    
mess_loss=tf.reduce_mean(tf.square(y_-cur_layers))

tf.add_to_collection('losses',mess_loss)

loss=tf.add_n(tf.get_collection('losses'))




import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)

print("Training data size:", mnist.train.num_examples)
print("Validating data size:", mnist.validation.num_examples)
print("Testing data size:", mnist.test.num_examples)

print("Example training data:", mnist.train.images[0])
print("Example training data label:", mnist.train.labels[0])

batch_size=100

xs,ys=mnist.train.next_batch(batch_size)
print("X shape:",xs.shape)
print("Y shape:",ys.shape)

INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500
BATCH_SIZE=100

LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99

REGULARIZATION_RATE=0.0001
TRAINGING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

def inference(input_tensor,avg_class,weight1,biases1,weight2,biases2):
    if avg_class==None:
        layer1=tf.nn.relu(tf.matmul(input_tensor,weight1)+biases1)
        return tf.matmul(layer1,weight2)+biases2
    else:
        layer1=tf.nn.relu(tf.matmul(input_tensor,avg_class.average(weight1))
        +avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weight2))+biases2
    
def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y_input')
    weight1=tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1))
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE]))
    weight2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    y=inference(x,None,weight1,biases1,weight2,biases2)
    
    global_step=tf.Variable(0,trainable=False)
    
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_average_op=variable_averages.apply(tf.trainable_variables())
    
    average__y=inference(x,variable_averages,weight1,biases1,weight2,biases2)
    
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    
    regularization=regularizer(weight1)+regularizer(weight2)
    
    loss=cross_entropy_mean+regularization
    
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,
                                             global_step,
                                             mnist.train.num_examples/BATCH_SIZE,
                                             LEARNING_RATE_DECAY)
    
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    with tf.control_dependencies([train_step,variables_average_op]):
        train_op=tf.no_op(name='train')
        
        
    correct_prediction=tf.equal(tf.argmax(average__y,1),tf.argmax(y_,1))
    
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        
        test_feed={x:mnist.test.images,y_:mnist.test.labels}
        
        for i in range(TRAINGING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,c_loss=sess.run([train_op,loss],feed_dict={x:xs,y_:ys})
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training steps,validation acc using average model is %g"%(i,validate_acc))
                print("loss=%g"%(c_loss))

        
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test acc using average model is %g"%(TRAINGING_STEPS,test_acc))
        


def main(argv=None):
    train(mnist)
    
    
main()




# coding=UTF-8 支持中文编码格式
import tensorflow as tf
import shutil
import os.path

MODEL_DIR = "d:/model/ckpt"
MODEL_NAME = "model.ckpt"

# if os.path.exists(MODEL_DIR): 删除目录
#     shutil.rmtree(MODEL_DIR)
if not tf.gfile.Exists(MODEL_DIR): #创建目录
    tf.gfile.MakeDirs(MODEL_DIR)

#下面的过程你可以替换成CNN、RNN等你想做的训练过程，这里只是简单的一个计算公式
input_holder = tf.placeholder(tf.float32, shape=[1], name="input_holder") #输入占位符，并指定名字，后续模型读取可能会用的
W1 = tf.Variable(tf.constant(5.0, shape=[1]), name="W1")
B1 = tf.Variable(tf.constant(1.0, shape=[1]), name="B1")
_y = (input_holder * W1) + B1
predictions = tf.greater(_y, 50, name="predictions") #输出节点名字，后续模型读取会用到，比50大返回true，否则返回false

init = tf.global_variables_initializer()
saver = tf.train.Saver() #声明saver用于保存模型

with tf.Session() as sess:
    sess.run(init)
    print ("predictions : ", sess.run(predictions, feed_dict={input_holder: [10.0]})) #输入一个数据测试一下
    saver.save(sess, os.path.join(MODEL_DIR, MODEL_NAME)) #模型保存
    print("%d ops in the final graph." % len(tf.get_default_graph().as_graph_def().node)) #得到当前图有几个操作节点

for op in tf.get_default_graph().get_operations(): #打印模型节点信息
    print (op.name, op.values())





import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)



import os
INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500


#下面这段代码有问题，不知道为什么运行不正常
def get_weight_variable(shape,regularizer):
    weights=tf.get_variable('weights',shape,tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(weights))
        
    return weights

def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weights=get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases=tf.get_variable('biases',[LAYER1_NODE],tf.float32,
                               initializer=tf.constant_initializer(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
        
    with tf.variable_scope('layer2'):
        weights=get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases=tf.get_variable('biases',[OUTPUT_NODE],
                               initializer=tf.constant_initializer(0.0))
        layer2=tf.nn.relu(tf.matmul(layer1,weights)+biases)
        
    return layer2


BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99

REGULARIZATION_RATE=0.0001
TRAINGING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

MODEL_SAVE_PATH='d:/path/model/'
MODEL_NAME='model.ckpt'

    
def train(mnist):
    x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y_input')
    
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    
    y=inference(x,regularizer)
    
    global_step=tf.Variable(0,trainable=False)
    
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_average_op=variable_averages.apply(tf.trainable_variables())
    
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    
    loss=cross_entropy_mean+tf.add_n(tf.get_collection('losses'))
    
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,
                                             global_step,
                                             mnist.train.num_examples/BATCH_SIZE,
                                             LEARNING_RATE_DECAY)
    
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    
    with tf.control_dependencies([train_step,variables_average_op]):
        train_op=tf.no_op(name='train')
        
        
    #持久化tensorflow类
    saver=tf.train.Saver()
    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        for i in range(TRAINGING_STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs,y_:ys})
            #每1000轮保存一次模型
            if i%1000==0:
                print("after %d training steps,loss on training batch is %g"%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)


def main(argv=None):
    train(mnist)
    
    
main()






import tensorflow as tf
import time

import os
INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500

BATCH_SIZE=100
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99

REGULARIZATION_RATE=0.0001
TRAINGING_STEPS=30000
MOVING_AVERAGE_DECAY=0.99

MODEL_SAVE_PATH='d:/path/model/'
MODEL_NAME='model.ckpt'

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)

EVAL_INTERVAL_SEC=10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
        y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y_input')
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        
        y=inference(x,None)
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)
        
        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                    print("after  training steps,valid acc using average model is %g"%(accuracy_score))
                    
            time.sleep(EVAL_INTERVAL_SEC)


evaluate(mnist)



import slim
with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,padding='SAME'):
    with tf.variable_scope('Mixes_7c'):
        branch_0=slim.conv2d(net,384,[1,1],scope='Conv2d_0a_1x1')
    with tf.variable_scope('Branch_1'):
        branch_1=tf.concat(3,[slim.conv2d(branch_1,384,[1,3],scope='Conv2d_0b_1x3'),
                              slim.conv2d(branch_1,384,[3,1],scope='Conv2d_0b_3x1')])

    
    