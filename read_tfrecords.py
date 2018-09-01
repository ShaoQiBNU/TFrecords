# -*- coding: utf-8 -*-
"""

读取TFRecords，将其打乱顺序批量输出

"""

##################### load packages #####################
import numpy as np
import os
from PIL import Image
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt


##################### read TFRecord and output batch #####################
def read_and_decode(filename, batch_size):

    '''
    filename: TFRecord路径
    '''

    ########### 根据文件名生成一个队列 ############
    filename_queue = tf.train.string_input_producer([filename])

    ########### 生成 TFRecord 读取器 ############
    reader = tf.TFRecordReader()
    
    ########### 返回文件名和文件 ############
    _, serialized_example = reader.read(filename_queue)

    ########### 取出example里的features #############
    features = tf.parse_single_example(serialized_example,
      features={
      'label': tf.FixedLenFeature([], tf.int64),
      'img' : tf.FixedLenFeature([], tf.string),
      'height': tf.FixedLenFeature([], tf.int64),
      'width': tf.FixedLenFeature([], tf.int64)})
    
    ########### 将序列化的img转为uint8的tensor #############
    img = tf.decode_raw(features['img'], tf.uint8)

    ########### 将label转为int32的tensor #############
    label = tf.cast(features['label'], tf.int32)

    ########### 将height和width转为int32的tensor #############
    height = tf.cast(features[ 'height' ], tf.int32)
    width = tf.cast(features[ 'width' ], tf.int32)
    
    ########### 将图片调整成正确的尺寸 ###########
    img = tf.reshape(img, [500, 500, 3])

    ########### 批量输出图片, 使用shuffle_batch可以有效地随机从训练数据中抽出batch_size个数据样本 ###########
    ##### shuffle batch之前，必须提前定义影像的size，size不可以是tensor，必须是明确的数字 ######
    ##### num_threads 表示可以选择用几个线程同时读取 #####
    ##### min_after_dequeue 表示读取一次之后队列至少需要剩下的样例数目 #####
    ##### capacity 表示队列的容量 #####
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size, capacity= 100, num_threads= 2, min_after_dequeue= 10)

    return img_batch, label_batch


########### tfrecords path ############
filename="/Users/shaoqi/Desktop/SPP/data/tfrecords/flower_train_500.tfrecords"

########### batch size ############
batch_size=2

########### get batch img and label ############
img_batch, label_batch = read_and_decode(filename, batch_size)


################### sess ######################
with tf.Session() as sess:

  ########### 初始化 ###########
  init = tf.global_variables_initializer()
  sess.run(init)

  ########## 启动队列线程 ##########
  coord=tf.train.Coordinator()
  threads= tf.train.start_queue_runners(sess=sess, coord=coord)

  ########### 取前3个batch查看 ###########
  for j in range(3):

    ######### 取出img_batch and label_batch #########
    img, label = sess.run([img_batch, label_batch])
    print(img.shape)

    ########## 显示每个batch的第一张图 ##########
    plt.imshow(img[0, :, :, :])
    plt.show()

  coord.request_stop()
  coord.join(threads)


