# -*- coding: utf-8 -*-
"""
preprocess flower data: 
           train:
           1. crop data, size exchange into 500 x 500
           2. save data and label into TFRecords

读取原始数据，

将train数据集裁剪成500 x 500，然后保存成TFRecords；

@author: shaoqi
"""

##################### load packages #####################
import numpy as np
import os
from PIL import Image
import scipy.io
import tensorflow as tf

##################### load flower data ##########################
def flower_preprocess(flower_folder):

    '''
    flower_floder: flower original path 原始花的路径
    flower_crop: 处理后的flower存放路径
    '''

    ######## flower dataset label 数据label ########
    labels = scipy.io.loadmat('/Users/shaoqi/Desktop/SPP/data/imagelabels.mat')
    labels = np.array(labels['labels'][0])-1


    ######## flower dataset: train test valid 数据id标识 ########
    setid = scipy.io.loadmat('/Users/shaoqi/Desktop/SPP/data/setid.mat')
    train = np.array(setid['tstid'][0]) - 1


    ######## flower data TFRecords save path TFRecords保存路径 ########
    writer_500 = tf.python_io.TFRecordWriter("/Users/shaoqi/Desktop/SPP/data/tfrecords/flower_train_500.tfrecords") 

    ######## flower data path 数据保存路径 ########
    flower_dir = list()


    ######## flower data dirs 生成保存数据的绝对路径和名称 ########
    for img in os.listdir(flower_folder):
        
        ######## flower data ########
        flower_dir.append(os.path.join(flower_folder, img))

    ######## flower data dirs sort 数据的绝对路径和名称排序 从小到大 ########
    flower_dir.sort()

    ###################### flower train data ##################### 
    for tid in train:
        ######## open image and get label ########
        img=Image.open(flower_dir[tid])
        
        ######## get width and height ########
        width,height=img.size

        ######## crop paramater ########
        h=500
        x=int((width-h)/2)
        y=int((height-h)/2)


        ################### crop image 500 x 500 and save image ##################
        img_crop=img.crop([x,y,x+h,y+h])

        ######## img to bytes 将图片转化为二进制格式 ########
        img_500=img_crop.tobytes()

        ######## build features 建立包含多个Features 的 Example ########
        example_500 = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[tid]])),
            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_500])),
            'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[500])),
            'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[500]))
            }))

        ######## 序列化为字符串,写入到硬盘 ########
        writer_500.write(example_500.SerializeToString())


################ main函数入口 ##################
if __name__ == '__main__':

    ######### flower path 鲜花数据存放路径 ########
    flower_folder = '/Users/shaoqi/Desktop/SPP/data/102flowers'
    
    ######## 数据预处理 ########
    flower_preprocess(flower_folder)