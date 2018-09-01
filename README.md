Tensorflow 处理自己的数据集
==================================
# 一. 数据集处理方法
> Tensorflow做深度学习时，难免会用到自己的数据，而且需要分batch做训练。处理方式有两种，一种是直接读入，利用Queue构建一个大小为capacity的缓存区，多线程执行数据的enqueue，神经网络模型从缓存区dequeue数据；另一种是采用Tensorflow的TFRecords格式，将数据提前处理成TFRecords格式，然后用queue读取，之后采用tf.train.shuffle_batch做训练。

# 二. TFrecords介绍
> TFrecords是一种二进制文件，它可以将资料与对应的资料标示（label）储存在一起，方便在TensorFlow中使用，MNIST等数据集都是采用此类格式存储的。

## (一) 写入TFrecords
### 1. 数据
> 数据集采用是102类的鲜花数据集，链接为：http://www.robots.ox.ac.uk/%7Evgg/data/flowers/102/.， 数据格式为jpg，数据集里有setid.mat参数文件，此文件是一个字典数据，其中包含三个key：trnid，tstid，valid，tstid为训练集id，trnid为测试集id，valid为验证集id。数据size不全都一样，因此将训练集中的影像全部做crop处理，裁剪成500 x 500大小的影像，然后制作成TFrecords文件。

### 2. Feature
> 标准的作法是将所有的变量先包装成Feature，然后将相关的Features（例如图片资料、标示等）组成一个Example，最后再将所有的Examples 存入TFRecords 中。包装Feature有一些基本小函数，代码如下：
```python
import tensorflow as tf

#二进制
def  _bytes_feature (value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

#整数
def  _int64_feature (value):
   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

#浮点数
def  _float32_feature (value):
   return tf.train.Feature(float_list=tf.train.FloatList(value=value))

```

### 3. 写入文件
> 根据变量的类型，将变量包装成Feature，再组成Example，然后写入TFRecords 中，代码如下：
```python
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
    writer_500 = tf.python_io.TFRecordWriter("/Users/shaoqi/Desktop/SPP/data/flower_train_500.tfrecords") 

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
```

## (二) 读取TFrecords

## 1. 普通测试，全部读取

> 建立好TFrecords后，可通过下列程序查看结果，测试写入是否正确，该方法会查看TFrecords里的所有对象，代码如下：

```python
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
    height = int (example.features.feature['height'].int64_list.value[0])
    width = int (example.features.feature['width'].int64_list.value[0]) 

    ######## get image and label ########
    image = (example.features.feature['img'].bytes_list.value[0]) 
    label = (example.features.feature['label'].int64_list.value[0])

    ######## img从string变成unint8 ########
    img = np.fromstring(image, dtype=np.uint8)

    ######## img reshape ########
    img = img.reshape((height, width, 3))

    ######## 显示影像 ########
    #plt.imshow(img)
    #plt.show()

################ main函数入口 ##################
if __name__ == '__main__':

  ########### tfrecords path ############
  filename="/Users/shaoqi/Desktop/SPP/data/flower_train_500.tfrecords"

  ######## read and check ########
  read_and_check(filename)
```

## 2. 利用队列高效读取

> 高效的读取方法应该是一个线程专门读取数据，一个线程专门做训练（前向反向传播），读取数据的线程应该维护一个队列(queue)，不断读取数据，压入队列，tensorflow里面常用的是FIFO queue，训练的线程每次从这个队列里面读取一个batch的训练数据用来训练。可以使用tf.string_input_produce创建上述的队列(filename queue)，然后通过读取文件名队列的文件名，进行解析，将解析得到的训练样例压入训练样例队列(example queue)。最后进行训练的时候可以使用tf.train.shuffle_batch来获取一个随机打乱顺序的batch，代码如下：

```python
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
    img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=batch_size, capacity= 1000, num_threads= 1, min_after_dequeue= 1000)

    return img_batch, label_batch


########### tfrecords path ############
filename="/Users/shaoqi/Desktop/SPP/data/flower_train_500.tfrecords"

########### batch size ############
batch_size=10

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
```
