#coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path='./mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path='./mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train='./data/mnist_train.tfrecords'
image_test_path='./mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path='./mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test='./data/mnist_test.tfrecords'
data_path='./data'
resize_height = 28
resize_width = 28

def write_tfRecord(tfRecordName, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)  
    num_pic = 0 
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()
    for content in contents:
        value = content.split()
        img_path = image_path + value[0] 
        img = Image.open(img_path)
        img_raw = img.tobytes() 
        labels = [0] * 10  
        labels[int(value[1])] = 1  
            
        example = tf.train.Example(features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
                })) 
        writer.write(example.SerializeToString())
        num_pic += 1 
        print ("the number of picture:", num_pic)
    writer.close()
    print("write tfrecord successful")

def generate_tfRecord():
	isExists = os.path.exists(data_path) 
	if not isExists: 
 		os.makedirs(data_path)
		print 'The directory was created successfully'
	else:
		print 'directory already exists' 
	write_tfRecord(tfRecord_train, image_train_path, label_train_path)
 	write_tfRecord(tfRecord_test, image_test_path, label_test_path)
  
def read_tfRecord(tfRecord_path):
    filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,
                                       features={
                                        'label': tf.FixedLenFeature([10], tf.int64),
                                        'img_raw': tf.FixedLenFeature([], tf.string)
                                        })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img.set_shape([784])
    img = tf.cast(img, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.float32)
    return img, label 
      
def get_tfrecord(num, isTrain=True):
    if isTrain:
        tfRecord_path = tfRecord_train
    else:
        tfRecord_path = tfRecord_test
    img, label = read_tfRecord(tfRecord_path)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size = num,
                                                    num_threads = 2,
                                                    capacity = 1000,
                                                    min_after_dequeue = 700)
    return img_batch, label_batch

def main():
    generate_tfRecord()

if __name__ == '__main__':
    main()
