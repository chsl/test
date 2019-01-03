# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 18:50:42 2018

@author: chsl-dxq
"""

#TFRecod

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data

def make_example(image, label):
    return tf.train.Example(features=tf.train.Features(feature={
        'image' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
        'label' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[label]))
    }))

def write_tfrecord(images, labels, filename):
    writer = tf.python_io.TFRecordWriter(filename)
    for image, label in zip(images, labels):
        labels = labels.astype(np.float32)
        ex = make_example(image.tobytes(), label.tobytes())
        writer.write(ex.SerializeToString())
    writer.close()

def main():
    fashion_mnist = input_data.read_data_sets('fashion', one_hot=True)
    train_images  = fashion_mnist.train.images
    train_labels  = fashion_mnist.train.labels
    test_images   = fashion_mnist.test.images
    test_labels   = fashion_mnist.test.labels
    write_tfrecord(train_images, train_labels, 'fashion_mnist_train.tfrecord')
    write_tfrecord(test_images, test_labels, 'fashion_mnist_test.tfrecord')



def read_tfrecord(filename):
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image'], tf.float32)
    label = tf.decode_raw(features['label'], tf.float64)

    image = tf.reshape(image, [28, 28, 1])
    label = tf.reshape(label, [10])

    image, label = tf.train.batch([image, label],
            batch_size=16,
            capacity=500)

    return image, label


def model(image, label):
    net = slim.conv2d(image, 48, [5,5], scope='conv1')
    net = slim.max_pool2d(net, [2,2], scope='pool1')
    net = slim.conv2d(net, 96, [5,5], scope='conv2')
    net = slim.max_pool2d(net, [2,2], scope='pool2')
    net = slim.flatten(net, scope='flatten')
    net = slim.fully_connected(net, 512, scope='fully_connected1')
    logits = slim.fully_connected(net, 10,
            activation_fn=None, scope='fully_connected2')

    prob = slim.softmax(logits)
    loss = slim.losses.softmax_cross_entropy(logits, label)

    train_op = slim.optimize_loss(loss, slim.get_global_step(),
            learning_rate=0.001,
            optimizer='Adam')

    return train_op

def main():
    train_images, train_labels =read_tfrecord('data/fashion_mnist_train.tfrecord')
    train_op = model(train_images, train_labels)

    step = 0
    with tf.Session() as sess:
        init_op = tf.group(
            tf.local_variables_initializer(),
            tf.global_variables_initializer())
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        while step < 3000:
            sess.run([train_op])

            if step % 100 == 0:
                print('step: {}'.format(step))

            step += 1

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
