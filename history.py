# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Fri Aug 31 20:02:16 2018)---
runfile('C:/Users/chsl-dxq/.spyder-py3/temp.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)
import tensorflow as tf
sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,784})
x=tf.placeholder(tf.float32,[None,784])
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))



y=tf.nn.softmax(tf.matmul(x,w)+b)
y=tf.nn.softmax(tf.matmul(x,W)+b)
help tf.matmul
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
print a
print(a)
runfile('C:/Users/chsl-dxq/.spyder-py3/temp.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
print(accuracy.eval({x:mnist.test.images,y_:mnist.test.labels,keep_prop:1.0}))
for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(128)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prop:0.75})
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys,keep_prop:0.75})
    loss=sess.run(cross_entroy,feed_dict={x:mnist.train.images,y:mnist.train.labels})
    print(str(i),"=",str(loss))
    
for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(128)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prop:0.75})
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys,keep_prop:0.75})
    loss=sess.run(cross_entroy,feed_dict={x:mnist.train.images,y_:mnist.train.labels,
                                          keep_prop:1.0})
    print(str(i),"=",str(loss))
                                          
tf.global_variables_initializer().run()
tf.global_variables_initializer().run()
for i in range(3000):
    batch_xs,batch_ys=mnist.train.next_batch(128)
    train_step.run({x:batch_xs,y_:batch_ys,keep_prop:0.75})
    sess.run(train_step,feed_dict={x:batch_xs,y_:batch_ys,keep_prop:0.75})
    loss=sess.run(cross_entroy,feed_dict={x:mnist.train.images,y_:mnist.train.labels,
                                          keep_prop:1.0})
    print(str(i),"=",str(loss))
                                          
runfile('C:/Users/chsl-dxq/.spyder-py3/temp.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
test_accuracy=accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prop:1.0})
print("step %d, train_accuracy %g"%test_accuracy )



## ---(Sat Nov 10 16:00:39 2018)---
runfile('C:/Users/chsl-dxq/untitled0.py', wdir='C:/Users/chsl-dxq')
for each_m in movies:
    print(each_m)
isinstance(movies,list)
dir(__builtins__)
for i in range(5):
    print(i)
data=open("d:/result.txt")
print(data.readline())
print(data.readline())
data.seek(0)
data.lineno(0)
data.seek(0)
for each_line in data:
    print(each_line)
data.close()
data=open("d:/result.txt")
print(data.readline())
print(data.readline())

data.seek(0)

for each_line in data:
    print(each_line.split(' '))
for each_line in data:
    print(each_line.split())
data.seek(0)

for each_line in data:
    print(each_line.split())
data=open("d:/result1.txt",'w')#写数据

print('test',file=data)

data.close()
try:
    data=open("d:/result1.txt",'w')#写数据
    print('test',file=data)
except IOError as err:
    print('file error'+str(err))
finally:
    if 'data' in locals:
        data.close()
try:
    data=open("d:/result1.txt",'w')#写数据
    print('test',file=data)
except IOError as err:
    print('file error'+str(err))
finally:
    if 'data' in locals():
        data.close()
import pickle

with open('mydata.pickle','wb') as mydata:
    pickle.dump([1,2,'txt'],mydata)


with open('mydata.pickle','rb') as reader:
    a_list=pickle.load(reader)


print(a_list)
data=[2,5,1,3,3,5,4]
data
data.sort()
data
data1=sorted(data)
data1
data2=[m*10 for m in data]
data2
type(distance)
distance=set()#第一种创建方法
distance={10,20,30,10}#第二种创建方法，大括号包围
type(distance)
distance={}
type(distance)
distance=set()#第一种创建方法
#distance={10,20,30,10}#第二种创建方法，大括号包围
distance={}
type(distance)
distance{'1'}=1
distance{'hehe'}=3
distance{'hehe'}=3
distance['hehe']=3
distance1=diict()
distance2={'1':1,'2':[2,3,4]}
distance['hehe']=3
distance1=dict()
distance2={'1':1,'2':[2,3,4]}
distance['hehe']=3
import numpy as np
a=np.array([1,2,3])
print(a)
a=np.array([[1,2],[3,4]])
print(a)
a=np.array([1,2,3])
a.ndim
a.ndim
a.dtype
help np.ones
help np.ones()
help np
help(np.ones)
np.ones([2,2])
np.ones(2,2)
np.ones((2,2))
help(np.random.rand)
help np.random.normal
help(np.random.normal)
arr = np.random.normal(1.75, 0.1, (4, 5))
print(arr)
after_arr = arr[1:3, 2:4]
print(after_arr)
after_arr = arr[1:2, 2:3]
after_arr
after_arr = arr[1:1, 2:3]
after_arr
stus_score = np.array([[80, 88], [82, 81], [84, 75], [86, 83], [75, 81]])
stus_score > 80
print(stus_score)
print(stus_score > 80)
print(stus_score)
help(np.where)
np.where(stus_score < 80, 0, 90)
print("每一列的最大值为:")
result = np.amax(stus_score, axis=0)
print(result)

print("每一行的最大值为:")
result = np.amax(stus_score, axis=1)
print(result)
print("每一列的最大值为:")
result = np.amax(stus_score, axis=0)
print(result)


# 求每一行的最小值(0表示列)
print("每一列的最小值为:")
result = np.amin(stus_score, axis=0)
print(result)


# 求每一行的平均值(0表示列)
print("每一列的平均值:")
result = np.mean(stus_score, axis=0)
print(result)

# 求每一行的方差(0表示列)
print("每一列的方差:")
result = np.std(stus_score, axis=0)
print(result)
help(np.amax)
result = np.amax(stus_score, axis=0,keepdims=True)
print(result)
result = np.amax(stus_score, axis=0,keepdims=False)
print(result)
q = np.array([[0.4], [0.6]])
result = np.dot(stus_score, q)
print("最终结果为:")
print(result)
help(tf.Variable)
from datatime import datatime
import math
import time
import tensorflow as tf
help(tf.Variable)
from datatime import datatime
import math
import time
import tensorflow as tf

batch_size=32
num_batches=100

def print_activations(t):
    print(t.op.name,' ',t.get_shape().as_list())



def interence(images):
    parameters=[]
    
    #第1个卷积层
    with tf.name_scope('conv1') as scope:
        kernel=tf.Variable(tf.truncated_normal([11,11,3,64],dtype=float32,
                                               stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(images,kernel,[1,4,4,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=float32),
                           trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv1=tf.nn.relu(bias,name=scope)
        lrn1=tf.nn.lrn(conv1,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn1')
        pool1=tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],
                             padding='VALID',name='pool1')
        parameters+=[kernel,biases]
        print_activations(conv1)
    
    
    #第2个卷积层    
    with tf.name_scope('conv2') as scope:
        kernel=tf.Variable(tf.truncated_normal([5,5,64,192],dtype=tf.float32,
                                               stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[64],dtype=tf.float32),
                                       trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv2=tf.nn.relu(bias)
        lrn2=tf.nn.lrn(conv2,4,bias=1.0,alpha=0.001/9,beta=0.75,name='lrn2')
        pool2=tf.nn.max_pool(lrn2,ksize=[1,3,3,1],strides=[1,2,2,1],
                             padding='VALID',name='pool2')
        parameters+=[kernel,biases]
        print_activations(conv2)
    
    #第3个卷积层
    with tf.name_scope('conv3') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,192,394],dtype=tf.float32,
                                               stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(pool2,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,shape=[384],dtype=tf.float32),
                           trainable=True,name=biases)
        bias=tf.nn.bias_add(conv,biases)
        conv3=tf.nn.relu(bias,name=scope)
        parameters+=[kernel,biases]
        print_activations(conv3)
    
    #第4个卷积层    
    with tf.name_scope('conv4') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,384,256],dtype=tf.float32,
                                               stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),
                           trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv4=tf.nn.relu(conv,biases)
        parameters+=[kernel,biases]
        print_activations(conv4)
    
    #第5个卷积层    
    with tf.name_scope('conv5') as scope:
        kernel=tf.Variable(tf.truncated_normal([3,3,256,256],dtype=tf.float32,
                                               stddev=1e-1),name='weights')
        conv=tf.nn.conv2d(conv3,kernel,[1,1,1,1],padding='SAME')
        biases=tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[256]),
                           trainable=True,name='biases')
        bias=tf.nn.bias_add(conv,biases)
        conv5=tf.nn.relu(conv,biases)
        parameters+=[kernel,biases]
        print_activations(conv5)
        pool5=tf.nn.max_pool(conv5,[1,3,3,1],strides=[1,2,2,1],padding='VALID',
                             name='pool5')
        print_activations(pool5)
    
    return pool5,parameters


def time_tensorflow_run(session,target,info_string):
    num_steps_burn_in=10
    total_duration=0.0
    total_duration_squared=0.0
    for i in range(num_batches+num_steps_burn_in):
        start_time=time.time()
        _=session.run(target)
        duration=time.time()-start_time
        if i>num_steps_burn_in:
            if not i%10:
                print('%s:step %d,duration=%.3f' % (datetime.now(),
                                                    i-num_steps_burn_in,duration))
            
            total_duration+=duration
            total_duration_squared=duration*duration
    
    mn=total_duration/num_batches
    vr=total_duration_squared/num_batches-mn*mn
    sd=math.sqrt(vr)
    print('%s:%s across %d steps,%.3f+/-%.3f sec /batch' % (datetime.now(),
                                                            info_string,
                                                            num_batches,
                                                            mn,sd))



def run_benchmark():
    with tf.Graph().as_default():
        image_size=224
        images=tf.Variable(tf.random_normal([batch_size,
                                             image_size,
                                             image_size,
                                             3],dtype=tf.float32,stddev=1e-1))
        pool5,parameters=interence(images)
        
        init=tf.global_variables_initializer()
        sess=tf.Session()
        sess.run(init)
        time_tensorflow_run(sess,pool5,'Forward')
        
        objective=tf.nn.l2_loss(pool5)
        grad=tf.gradients(objective,parameters)
        time_tensorflow_run(sess,grad,'Forward_backward')


run_benchmark()
runfile('C:/Users/chsl-dxq/.spyder-py3/mnist.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
debugfile('C:/Users/chsl-dxq/.spyder-py3/mnist.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
runfile('C:/Users/chsl-dxq/.spyder-py3/mnist.py', wdir='C:/Users/chsl-dxq/.spyder-py3')

## ---(Tue Nov 13 08:55:41 2018)---
runfile('C:/Users/chsl-dxq/.spyder-py3/mnist.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
for i in range(5):
    print(i)
runfile('C:/Users/chsl-dxq/.spyder-py3/mnist.py', wdir='C:/Users/chsl-dxq/.spyder-py3')

## ---(Wed Nov 14 19:05:47 2018)---
import tensorflow as tf

with tf.variable_scope('scope1'):
    s1=1;
    s2=2;


sess=tf.Session()
sess.run(s1)
sess.run(s2)
print(s1.name)
print(s2.name)
with tf.variable_scope('scope1'):
    s1=1;
    s2=2;


print(s1.name)
print(s2.name)
image_size=224
batch_size=24
images=tf.Variable(tf.random_normal([batch_size,
                                     image_size,
                                     image_size,
                                     3],dtype=tf.float32,stddev=1e-1))

inception_v3_base(images,scope='my')
def inception_v3_base(inputs,scope=None):
    end_points={}
    with tf.variable_scope(scope,'InceptionV3',[inputs]):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                            stride=1,padding='VALID'):
            net=slim.conv2d(inputs,32,[3,3],stride=2,scope='Conv2d_1a_3*3')
            print(net.name)
            net=slim.conv2d(net,32,[3,3],scope='Conv2d_2a_3*3')
            net=slim.conv2d(net,64,[3,3],padding='SAME',scope='Conv2d_2b_3*3')
            net=slim.max_pool2d(net,[3,3],stride=2,scope='MaxPool_3a_3*3')
            net=slim.conv2d(net,80,[1,1],scope='Conv2d_3b_1*1')
            net=slim.conv2d(net,192,[3,3],scope='Conv2d_4a_3*3')
            net=slim.max_pool2d(net,[3,3],stride=2,scope='MaxPool_5a_3*3')
        
        with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],
                            stride=1,padding='SAME'):
            with tf.variable_scope('Mixed_5b'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1*1')
                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1*1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5*5')
                with tf.variable_scope('Branch_2'):
                    branch_2=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1*1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3*3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3*3')
                with tf.variable_scope('Branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='AvfPool_0a_3*3')
                    branch_3=slim.conv2d(branch_3,32,[1,1],scope='Conv2d_0b_1*1')
            
            net=tf.concat([branch_0,branch_1,branch_2,branch_3])
            
            with tf.variable_scope('Mixed_5c'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1*1')
                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1*1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5*5')
                with tf.variable_scope('Branch_2'):
                    branch_2=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1*1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3*3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3*3')
                with tf.variable_scope('Branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='AvfPool_0a_3*3')
                    branch_3=slim.conv2d(branch_3,64,[1,1],scope='Conv2d_0b_1*1')
            
            net=tf.concat([branch_0,branch_1,branch_2,branch_3])
            
            with tf.variable_scope('Mixed_5d'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1*1')
                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,48,[1,1],scope='Conv2d_0a_1*1')
                    branch_1=slim.conv2d(branch_1,64,[5,5],scope='Conv2d_0b_5*5')
                with tf.variable_scope('Branch_2'):
                    branch_2=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1*1')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0b_3*3')
                    branch_2=slim.conv2d(branch_2,96,[3,3],scope='Conv2d_0c_3*3')
                with tf.variable_scope('Branch_3'):
                    branch_3=slim.avg_pool2d(net,[3,3],scope='AvfPool_0a_3*3')
                    branch_3=slim.conv2d(branch_3,64,[1,1],scope='Conv2d_0b_1*1')
                
                net=tf.concat([branch_0,branch_1,branch_2,branch_3])
            
            with tf.variable_scope('Mixed_6a'):
                with tf.variable_scope('Branch_0'):
                    branch_0=slim.conv2d(net,384,[3,3],stride=2,
                                         padding='VALID',scope='Conv2d_0a_3*3')
                with tf.variable_scope('Branch_1'):
                    branch_1=slim.conv2d(net,64,[1,1],scope='Conv2d_0a_1*1')
                    branch_1=slim.conv2d(branch_1,96,[3,3],scope='Conv2d_0b_3*3')
                    branch_1=slim.conv2d(branch_1,96,[3,3],stride=2,
                                         padding='VALID',scope='Conv2d_0c_3*3')
                with tf.variable_scope('Branch_2'):
                    branch_2=slim.max_pool2d(net,[3,3],stride=2,
                                             padding='VALID',scope='MaxPool_0a_3*3')
                
                net=tf.concat([branch_0,branch_1,branch_2])
            
            end_points['Mixed_6e']=net



image_size=224
batch_size=24
images=tf.Variable(tf.random_normal([batch_size,
                                     image_size,
                                     image_size,
                                     3],dtype=tf.float32,stddev=1e-1))

inception_v3_base(images,scope='my')
runfile('C:/Users/chsl-dxq/.spyder-py3/mnist.py', wdir='C:/Users/chsl-dxq/.spyder-py3')

## ---(Wed Nov 14 21:19:38 2018)---
import collections
import tensorflow as tf

input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)
inter=tf.add(input1,input2)
mul=tf.mul(input1,input2)

with tf.Session() as sess:
    result=sess.run([mul,inter])
    print(result)
mul=tf.matmul(input1,input2)

with tf.Session() as sess:
    result=sess.run([mul,inter])
    print(result)
mul=tf.sub(input1,input2)

with tf.Session() as sess:
    result=sess.run([mul,inter])
    print(result)
mul=tf.add(input3,input2)

with tf.Session() as sess:
    result=sess.run([mul,inter])
    print(result)
print(inter.name)
input1=tf.constant(3.0)
input2=tf.constant(2.0)
input3=tf.constant(5.0)

## ---(Thu Nov 15 13:35:05 2018)---
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3]) 
import tensorflow as tf
a = tf.constant([1, 2, 3, 4, 5, 6], shape=[2, 3])
a
print(a
)
sess=tf.Session()
sess.run(a)
a=tf.truncated_normal([2,3,4])
sess.run(a)
tf.reduce_max(a,0)
b=tf.truncated_normal([3,2,3,4])
tf.reduce_max(b,0)
print(b.name)
print(b.op.name)
tensorboard --logdir logs
import tensorflow as tf

with tf.name_scope('graph') as scope:
     matrix1 = tf.constant([[3., 3.]],name ='matrix1')  #1 row by 2 column
     matrix2 = tf.constant([[2.],[2.]],name ='matrix2') # 2 row by 1 column
     product = tf.matmul(matrix1, matrix2,name='product')


sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()

sess.run(init)
import tensorflow as tf

with tf.name_scope('graph') as scope:
     matrix1 = tf.constant([[3., 3.]],name ='matrix1')  #1 row by 2 column
     matrix2 = tf.constant([[2.],[2.]],name ='matrix2') # 2 row by 1 column
     product = tf.matmul(matrix1, matrix2,name='product')
  
sess = tf.Session()

writer = tf.summary.FileWriter("/", sess.graph)

init = tf.global_variables_initializer()

sess.run(init)


import tensorflow as tf

with tf.name_scope('graph') as scope:
     matrix1 = tf.constant([[3., 3.]],name ='matrix1')  #1 row by 2 column
     matrix2 = tf.constant([[2.],[2.]],name ='matrix2') # 2 row by 1 column
     product = tf.matmul(matrix1, matrix2,name='product')


sess = tf.Session()

writer = tf.summary.FileWriter("logs/", sess.graph)

init = tf.global_variables_initializer()

sess.run(init)
tensorboard --logdir logs
tensorboard --logdir logs
import tensorflow as tf

with tf.name_scope('graph') as scope:
     matrix1 = tf.constant([[3., 3.]],name ='matrix1')  #1 row by 2 column
     matrix2 = tf.constant([[2.],[2.]],name ='matrix2') # 2 row by 1 column
     product = tf.matmul(matrix1, matrix2,name='product')


sess = tf.Session()

writer = tf.summary.FileWriter("d:/", sess.graph)

init = tf.global_variables_initializer()

sess.run(init)
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
writer = tf.summary.FileWriter("d:\logs\", sess.graph)
sess.run(init)
## Loop
for step  in  range(101):
    sess.run(train)
    if step %10==0 :
        print step ,'weight:',sess.run(weight),'bias:',sess.run(bias)
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
    if step %10==0 :
        print step ,'weight:',sess.run(weight),'bias:',sess.run(bias)
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
    if step %10==0 :
        print step,'weight:',sess.run(weight),'bias:',sess.run(bias)
for step  in  range(101):
    sess.run(train)
    if step %10==0:
        print step,'weight:',sess.run(weight),'bias:',sess.run(bias)
for step  in  range(101):
    sess.run(train)
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
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
## Loop
for step  in  range(101):
    sess.run(train)
    rs=sess.run(merged)
    writer.add_summary(rs, step)
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
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
## Loop
for step  in  range(101):
    sess.run(train)
    rs=sess.run(merged)
    writer.add_summary(rs, step)
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
writer = tf.summary.FileWriter("d:\logs\", sess.graph)
sess.run(init)
## Loop
for step  in  range(101):
    sess.run(train)
    rs=sess.run(merged)
    writer.add_summary(rs, step)
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
writer = tf.summary.FileWriter('d:\logs\', sess.graph)
sess.run(init)
## Loop
for step  in  range(101):
    sess.run(train)
    rs=sess.run(merged)
    writer.add_summary(rs, step)
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

## ---(Wed Nov 21 15:41:00 2018)---
import tensorflow as tf

input1=tf.constant([1.0,2.0,3.0],name='input1')
input2=tf.Variable(tf.random_uniform([3]),name='input2')
output=tf.add_n([input1,input2],name='add')

writer=tf.train.SummaryWriter('d:/log',tf.get_default_graph())
writer.close()


import tensorflow as tf

input1=tf.constant([1.0,2.0,3.0],name='input1')
input2=tf.Variable(tf.random_uniform([3]),name='input2')
output=tf.add_n([input1,input2],name='add')

writer=tf.train.summary.FileWriter('d:/log',tf.get_default_graph())
writer.close();
import tensorflow as tf

input1=tf.constant([1.0,2.0,3.0],name='input1')
input2=tf.Variable(tf.random_uniform([3]),name='input2')
output=tf.add_n([input1,input2],name='add')

writer=tf.summary.FileWriter('d:/log',tf.get_default_graph())
writer.close();
with tf.Variabl_scope('foo'):
    a=tf.get_variable('bar',[1])
    print(a.name)



with tf.Variable_scope('bar'):
    b=tf.get_variable('bar',[2])
    print(b.name)


with tf.name_scope('a'):
    a=tf.Variable([3])
    print(a.name)
    
    a=tf.get_variable('b',[1])
    print(a.name)



with tf.name_scope('b'):
    tf.get_variable('b',[1])
with tf.variabl_scope('foo'):
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
with tf.name_scope('input1'):
    input1=tf.constant([1.0,2.0,3.0],name='input1')


with tf.name_scope('input2'):
    input2=tf.Variable(tf.random_uniform([3]),name='input2')

output=tf.add_n([input1,input2],name='add')

writer=tf.summary.FileWriter('d:/log',tf.get_default_graph())
writer.close();
import tensorflow as tf

with tf.name_scope('input1'):
    input1=tf.constant([1.0,2.0,3.0],name='input1')


with tf.name_scope('input2'):
    input2=tf.Variable(tf.random_uniform([3]),name='input2')

output=tf.add_n([input1,input2],name='add')

writer=tf.summary.FileWriter('d:/log',tf.get_default_graph())
writer.close();
import tensorflow as tf

print(tf.get_default_graph.collections)
g=tf.Graph()
print(g.collections)
g=tf.Graph()
with g.as_default():
    print(g.collections)
g=tf.Graph()
with g.as_default():
    a=tf.constant(1)
    print(g.collections)
g=tf.Graph()
with g.as_default():
    a=tf.constant(1)
    print(g.graph_def_versions)
g=tf.Graph()
with g.as_default():
    a=tf.constant(1)
    g.add_to_collection('tt',a)
    print(g.collections)
x=tf.constatnt([[0.7,0.9]])
x(0,0)=1;
print(x)
x=tf.constatnt([[0.7,0.9]])
tf.assign(x,[1.0,2.0])
ini=tf.global_variables_initializer()
with tf.session() as sess:
    sess.run(ini)
    print(x)
x=tf.constant([[0.7,0.9]])
tf.assign(x,[1.0,2.0])
ini=tf.global_variables_initializer()
with tf.session() as sess:
    sess.run(ini)
    print(x)
x=tf.constant([[0.7,0.9]])
#tf.assign(x,[1.0,2.0])
ini=tf.global_variables_initializer()
with tf.session() as sess:
    sess.run(ini)
    print(x)
x=tf.constant([[0.7,0.9]])
#tf.assign(x,[1.0,2.0])
ini=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(ini)
    print(x)
x=tf.constant([[0.7,0.9]])
#tf.assign(x,[1.0,2.0])
ini=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(ini)
    print(x.eval(sess))
x=tf.constant([[0.7,0.9]])
#tf.assign(x,[1.0,2.0])
ini=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(ini)
    print(sess.run(x))
for i inrange(1,10)
for i in range(1,10):
    print(i)
    

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)

print "Training data size:", mnist.train.num_examples
print "Validating data size:", mnist.validation.num_examples
print "Testing data size:", mnist.test.num_examples

print "Example training data:", mnist.train.images[0]
print "Example training data label:", mnist.train.labels[0]
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)

print("Training data size:", mnist.train.num_examples)
print("Validating data size:", mnist.validation.num_examples)
print("Testing data size:", mnist.test.num_examples)

print("Example training data:", mnist.train.images[0])
print("Example training data label:", mnist.train.labels[0])
print("Training data size:", mnist.train.num_examples)
print("Validating data size:", mnist.validation.num_examples)
print("Testing data size:", mnist.test.num_examples)

print("Example training data:", mnist.train.images[0])
print("Example training data label:", mnist.train.labels[0])
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)

print("Training data size:", mnist.train.num_examples)
print("Validating data size:", mnist.validation.num_examples)
print("Testing data size:", mnist.test.num_examples)

print("Example training data:", mnist.train.images[0])
print("Example training data label:", mnist.train.labels[0])
xs,ys=mnist.train.next_batch(batch_size)
print("X shape:",xs.shape)
print("Y shape:",ys.shape)
batch_size=100

xs,ys=mnist.train.next_batch(batch_size)
print("X shape:",xs.shape)
print("Y shape:",ys.shape)
tf.argmax([[1,2],[3,4]],1)
with tf.Session() as sess
with tf.Session() as sess:
    print(tf.argmax([[1,2],[3,4]],1))
with tf.Session() as sess:
    print(sess.run(tf.argmax([[1,2],[3,4]],1)))
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
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE])
    
    weight2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))
    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    y=inference(x,None,weight1,biases1,weight2,biases2)
    
    global_step=tf.Variable(0,trainable=False)
    
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_average_op=variable_averages.apply(tf.trainable_variables())
    
    average__y=inference(x,variable_averages,weight1,biases1,weight2,biases2)
    
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    
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
            if i%1000=0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training steps,validation acc using average model is %g"%(i,validate_acc))
            
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test acc using average model is %g"%(TRAINGING_STEPS,test_acc))




def main(argv=None):
    train(mnist)



main()
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
    biases1=tf.Variable(tf.constant(0.1,shape=[LAYER1_NODE])
    weight2=tf.Variable(tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1))

    biases2=tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))
    
    y=inference(x,None,weight1,biases1,weight2,biases2)
    
    global_step=tf.Variable(0,trainable=False)
    
    variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    variables_average_op=variable_averages.apply(tf.trainable_variables())
    
    average__y=inference(x,variable_averages,weight1,biases1,weight2,biases2)
    
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    
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
            if i%1000=0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training steps,validation acc using average model is %g"%(i,validate_acc))
                
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test acc using average model is %g"%(TRAINGING_STEPS,test_acc))
        


def main(argv=None):
    train(mnist)
    
    
main()
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
    
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    
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
            if i%1000=0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training steps,validation acc using average model is %g"%(i,validate_acc))
            
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test acc using average model is %g"%(TRAINGING_STEPS,test_acc))




def main(argv=None):
    train(mnist)



main()
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
    
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    
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
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training steps,validation acc using average model is %g"%(i,validate_acc))
            
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test acc using average model is %g"%(TRAINGING_STEPS,test_acc))




def main(argv=None):
    train(mnist)



main()
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
    
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(y,tf.argmax(y_,1))
    
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
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training steps,validation acc using average model is %g"%(i,validate_acc))
            
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test acc using average model is %g"%(TRAINGING_STEPS,test_acc))




def main(argv=None):
    train(mnist)



main()
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
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training steps,validation acc using average model is %g"%(i,validate_acc))
            
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test acc using average model is %g"%(TRAINGING_STEPS,test_acc))




def main(argv=None):
    train(mnist)



main()
x=[[1],[2]]
x.shape
x.size
x=tf.argmax([[1,2][3,4]],0)
x=tf.argmax([[1,2],[3,4]],0)
x
with tf.Session() as sess:
    print(sess.run(x))
 x=tf.argmax([[1,2],[3,4]],1)
with tf.Session() as sess:
    print(sess.run(x))
 x=tf.argmax([[1,2,3],[3,4,2]],1)
with tf.Session() as sess:
    print(sess.run(x))
 x=tf.argmax([[1,2,3],[3,4,2]],0)
with tf.Session() as sess:
    print(sess.run(x))
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
import os
INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500



def get_weight_variable(shape,regularizer):
    weights=tf.get_variable('weights',shape,tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1)))
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(weights))
    
    return weights


def inference(input_tensor,regularizer):
    with tf.variable_scope('layer1'):
        weights=get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases=tf.get_variable('biases',[LAYER1_NODE],tf.float32,
                               initializer=tf.constant(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    
    with tf.variable_scope('layer2'):
        weights=get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases=tf.get_variable('biases',[OUTPUT_NODE],
                               initializer=tf.constant(0.0))
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
    
    average__y=inference(x,variable_averages,weight1,biases1,weight2,biases2)
    
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
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)



def main(argv=None):
    train(mnist)



main()
import os
INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500



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
                               initializer=tf.constant(0.0))
        layer1=tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    
    with tf.variable_scope('layer2'):
        weights=get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases=tf.get_variable('biases',[OUTPUT_NODE],
                               initializer=tf.constant(0.0))
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
    
    average__y=inference(x,variable_averages,weight1,biases1,weight2,biases2)
    
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
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)



def main(argv=None):
    train(mnist)



main()
import os
INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500



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
    
    average__y=inference(x,variable_averages,weight1,biases1,weight2,biases2)
    
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
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)



def main(argv=None):
    train(mnist)



main()
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)



import os
INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500



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
    
    average__y=inference(x,variable_averages,weight1,biases1,weight2,biases2)
    
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
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)



def main(argv=None):
    train(mnist)
main()
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)



import os
INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500



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
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)



def main(argv=None):
    train(mnist)



main()
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)



import os
INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500



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

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)

EVAL_INTERVAL_SEC=10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
        y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y_input')
        validate_feed={x:mnist.validation.images,y_:mnist.validation.labels}
        
        y=inference(x,None)
        correct_prediction=tf.equal(tf.argmax(average__y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)
        
        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    accuracy_score=sess.run(accuracy,feed_dictd_dict=validate_feed)
                    print("after  training steps,valid acc using average model is %g"%(accuracy_score))
            
            time.sleep(EVAL_INTERVAL_SEC)



evaluate()
evaluate(mnist)
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
        correct_prediction=tf.equal(tf.argmax(average__y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)
        
        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    accuracy_score=sess.run(accuracy,feed_dictd_dict=validate_feed)
                    print("after  training steps,valid acc using average model is %g"%(accuracy_score))
            
            time.sleep(EVAL_INTERVAL_SEC)



evaluate(mnist)
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
evaluate(mnist)
def get_weight_variable(shape,regularizer):
    weights=tf.get_variable('weights',shape,tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection('losses',regularizer(weights))
    
    return weights
evaluate(mnist)
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
evaluate(mnist)
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
                    accuracy_score=sess.run(accuracy,feed_dictd_dict=validate_feed)
                    print("after  training steps,valid acc using average model is %g"%(accuracy_score))
            
            time.sleep(EVAL_INTERVAL_SEC)



evaluate(mnist)
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
                    accuracy_score=sess.run(accuracy,feed_dictd=validate_feed)
                    print("after  training steps,valid acc using average model is %g"%(accuracy_score))
            
            time.sleep(EVAL_INTERVAL_SEC)



evaluate(mnist)
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
                print(sess.run(variables_to_restore))
                ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                    print("after  training steps,valid acc using average model is %g"%(accuracy_score))
            
            time.sleep(EVAL_INTERVAL_SEC)



evaluate(mnist)
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
        
        while True:
            with tf.Session() as sess:
                print(sess.run(variables_to_restore))
                ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                    print("after  training steps,valid acc using average model is %g"%(accuracy_score))
            
            time.sleep(EVAL_INTERVAL_SEC)



evaluate(mnist)
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
        
        while True:
            with tf.Session() as sess:
                ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    accuracy_score=sess.run(accuracy,feed_dict=validate_feed)
                    print("after  training steps,valid acc using average model is %g"%(accuracy_score))
            
            time.sleep(EVAL_INTERVAL_SEC)



evaluate(mnist)
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
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)



import os
INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500



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
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training steps,validation acc using average model is %g"%(i,validate_acc))
                print("loss=%g"%(sess.run(loss)))
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test acc using average model is %g"%(TRAINGING_STEPS,test_acc))




def main(argv=None):
    train(mnist)



main()
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
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training steps,validation acc using average model is %g"%(i,validate_acc))
                print("loss=%g"%(sess.run(loss,feed_dict={x:xs,y_:ys})))
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op,feed_dict={x:xs,y_:ys})
        
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test acc using average model is %g"%(TRAINGING_STEPS,test_acc))




def main(argv=None):
    train(mnist)



main()
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
            loss,_=sess.run(train_op,feed_dict={x:xs,y_:ys})
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training steps,validation acc using average model is %g"%(i,validate_acc))
                print("loss=%g"%(loss))
        
        
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test acc using average model is %g"%(TRAINGING_STEPS,test_acc))




def main(argv=None):
    train(mnist)



main()
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
            _,c_loss=sess.run(train_op,loss,feed_dict={x:xs,y_:ys})
            if i%1000==0:
                validate_acc=sess.run(accuracy,feed_dict=validate_feed)
                print("after %d training steps,validation acc using average model is %g"%(i,validate_acc))
                print("loss=%g"%(c_loss))
        
        
        test_acc=sess.run(accuracy,feed_dict=test_feed)
        print("after %d training steps,test acc using average model is %g"%(TRAINGING_STEPS,test_acc))




def main(argv=None):
    train(mnist)



main()
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
import tensorflow as tf


from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MINIST_data/",one_hot=True)



import os
INPUT_NODE=784
OUTPUT_NODE=10

LAYER1_NODE=500



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

curl http://download.tensorflow.org/example images/flower photos.tgz
tar xzf flower_photos.tgz

wget https://storage.googleapis.com/download.tensorflow.org/models/\inception_dec_2015.zip
unzip tensorflow/examples/label_image/data/inception_dec_2015.zip
curl http://download.tensorflow.org/example_images/flower_photos.tgz
tar xzf flower_photos.tgz
import tensorflow as tf

#curl http://download.tensorflow.org/example_images/flower_photos.tgz
#tar xzf flower_photos.tgz

#wget https://storage.googleapis.com/download.tensorflow.org/models/\inception_dec_2015.zip
#unzip tensorflow/examples/label_image/data/inception_dec_2015.zip

import glob
import os.path
import random
import numpy as np
from tensorflow.python.platform import gfile

BOTTLENECT_TENSOR_SIZE=2048

BOTTLENECT_TENSOR_NAME='pool_3/reshape:0'

JPEG_DATA_TENSOR_NAME='DecodeJpeg/contents:0'

MODEL_DIR='/inception_v3'

MODEL_FILE='tensorflow_inception_graph.pd'

CACHE_DIR='/inception_v3'

INPUT_DATA='/flower_photos'

VALIDATION_PERCENTAGE=10

TEST_PERCENTAGE=10

LEARNING_RATE=0.01
STEPS=4000
BATCH=100

def create_image_lists(testing_percentage,validation_percentage):
    result={}
    sub_dirs=[x[0] for x in os.walk(INPUT_DATA)]
    is_root_dir=True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir=False
            continue
        extensions=['jpg','jpeg','JPG','JPEG']
        file_list=[]
        dir_name=os.path.basename(sub_dir)
        for extension in extensions:
            file_glob=os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue
        
        label_name=dir_name.lower()
        traing_images=[]
        testing_images=[]
        validation_images=[]
        for file_name in file_list:
            base_name=os.path.basename(file_name)
            chance=np.random.randint(100)
            if chance<validation_percentage:
                validation_images.append(base_name)
            elif chance<(testing_percentage+validation_percentage):
                testing_images.append(base_name)
            else:
                traing_images.append(base_name)
        
        result[label_name]={
                'dir':dir_name,
                'trainng':traing_images,
                'testing':testing_images,
                'validation':validation_images}
    
    return result



def get_image_path(image_lists,image_dir,label_name,index,category):
    label_lists=image_lists[label_name]
    category_list=label_lists[category]
    mod_index=index%len(category_list)
    base_name=category_list[mod_index]
    sub_dir=label_lists['dir']
    full_path=os.path.join(image_dir,sub_dir,base_name)
    return full_path



result=create_image_lists(10,10)
debugfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
str=get_bottleneck_path(result,'daisy',2,'testing')
def get_image_path(image_lists,image_dir,label_name,index,category):
    label_lists=image_lists[label_name]
    category_list=label_lists[category]
    mod_index=index%len(category_list)
    base_name=category_list[mod_index]
    sub_dir=label_lists['dir']
    full_path=os.path.join(image_dir,sub_dir,base_name)
    return full_path



def get_bottleneck_path(image_lists,label_name,index,category):
    return get_image_path(image_lists,CACHE_DIR,label_name,index,category)+'.txt'



str=get_bottleneck_path(result,'daisy',2,'testing')
runfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
debugfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
runfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
debugfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
runfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
debugfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
runfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
debugfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
runfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
debugfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
runfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
debugfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
runfile('C:/Users/chsl-dxq/.spyder-py3/transferLearning.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
runfile('C:/Users/chsl-dxq/.spyder-py3/TFRecod.py', wdir='C:/Users/chsl-dxq/.spyder-py3')
debugfile('C:/Users/chsl-dxq/.spyder-py3/TFRecod.py', wdir='C:/Users/chsl-dxq/.spyder-py3')

## ---(Thu Nov 29 21:24:18 2018)---
runfile('C:/Users/chsl-dxq/.spyder-py3/untitled1.py', wdir='C:/Users/chsl-dxq/.spyder-py3')