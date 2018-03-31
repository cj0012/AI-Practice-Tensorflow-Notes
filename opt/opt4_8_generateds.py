#coding:utf-8
#0导入模块 ，生成模拟数据集
import numpy as np
import matplotlib.pyplot as plt
seed = 2 
def generateds():
	#基于seed产生随机数
	rdm = np.random.RandomState(seed)
	#随机数返回300行2列的矩阵，表示300组坐标点（x0,x1）作为输入数据集
	X = rdm.randn(300,2)
	#从X这个300行2列的矩阵中取出一行,判断如果两个坐标的平方和小于2，给Y赋值1，其余赋值0
	#作为输入数据集的标签（正确答案）
	Y_ = [int(x0*x0 + x1*x1 <2) for (x0,x1) in X]
	#遍历Y中的每个元素，1赋值'red'其余赋值'blue'，这样可视化显示时人可以直观区分
	Y_c = [['red' if y else 'blue'] for y in Y_]
	#对数据集X和标签Y进行形状整理，第一个元素为-1表示跟随第二列计算，第二个元素表示多少列，可见X为两列，Y为1列
	X = np.vstack(X).reshape(-1,2)
	Y_ = np.vstack(Y_).reshape(-1,1)
	
	return X, Y_, Y_c
	
#print X
#print Y_
#print Y_c
#用plt.scatter画出数据集X各行中第0列元素和第1列元素的点即各行的（x0，x1），用各行Y_c对应的值表示颜色（c是color的缩写） 
#plt.scatter(X[:,0], X[:,1], c=np.squeeze(Y_c)) 
#plt.show()
