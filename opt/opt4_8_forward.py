#coding:utf-8
#0导入模块 ，生成模拟数据集
import tensorflow as tf

#定义神经网络的输入、参数和输出，定义前向传播过程 
def get_weight(shape, regularizer):
	w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
	tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):  
    b = tf.Variable(tf.constant(0.01, shape=shape)) 
    return b
	
def forward(x, regularizer):
	
	w1 = get_weight([2,11], regularizer)	
	b1 = get_bias([11])
	y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = get_weight([11,1], regularizer)
	b2 = get_bias([1])
	y = tf.matmul(y1, w2) + b2 
	
	return y
