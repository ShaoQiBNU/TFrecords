# -*- coding: utf-8 -*-
"""

读取TFRecords，查看写入是否正确

"""

##################### load packages #####################
import numpy as np
import os
from PIL import Image
import scipy.io
import tensorflow as tf
import matplotlib.pyplot as plt

################ read tfrecords and check ##################
def read_and_check(filename):

  '''
  filename: TFRecord路径
  '''

  ################ read tfrecords 读取tfrecords ##################
  record_iterator = tf.python_io.tf_record_iterator(path=filename)

  ################ 读取tfrecords里的每一个对象，即写入的所有对象 ##################
  for string_record in record_iterator:

    ######## 建立example ########
    example = tf.train.Example()

    ######## 解析TFRecords 里的feature ########
    example.ParseFromString(string_record)

    ######## get height and width ########
    height = int(example.features.feature['height'].int64_list.value[0])
    width = int(example.features.feature['width'].int64_list.value[0]) 

    ######## get image and label ########
    image = (example.features.feature['img'].bytes_list.value[0]) 
    label = (example.features.feature['label'].int64_list.value[0])

    ######## img从string变成unint8 ########
    img = np.fromstring(image, dtype=np.uint8)

    ######## img reshape ########
    img = img.reshape((height, width, 3))

    ######## 显示影像 ########
    plt.imshow(img)
    plt.show()


################ main函数入口 ##################
if __name__ == '__main__':

  ########### tfrecords path ############
  filename="/Users/shaoqi/Desktop/SPP/data/tfrecords/flower_train_500.tfrecords"

  ######## read and check ########
  read_and_check(filename)

