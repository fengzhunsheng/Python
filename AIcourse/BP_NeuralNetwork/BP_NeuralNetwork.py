# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 18:54:37 2019
@file:  BP_NeuralNetwork
@brief: use BP_NN to classify 4*10*3

@author: 冯准生
@email:1565853379@qq.com
"""

import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import numpy as np
import time


def neuralNetwork(input_layer_size,hidden_layer_size,out_put_layer,max_iter):
    #对文件进行处理，使得成为便于计算的numpy数据类型
    data_source_1 = pd.read_csv("Iris-train.txt", sep='[ |\t]',
                        names=['f1','f2','f3','f4','labels'],engine='python')
    data_train = np.array(data_source_1)
    X_train = data_train[:,:4]
    y_train = data_train[:,4]     

    data_source_2 = pd.read_csv("Iris-test.txt", sep='[ |\t]',
                        names=['f1','f2','f3','f4','labels'],engine='python')
    data_test = np.array(data_source_2)
    X_test = data_test[:,:4]
    y_test = data_test[:,4]
    
    Lambda = 1
    
    initial_Theta1 = randInitializeWeights(input_layer_size,hidden_layer_size) 
    initial_Theta2 = randInitializeWeights(hidden_layer_size,out_put_layer)
    
    initial_nn_params = np.vstack((initial_Theta1.reshape(-1,1),initial_Theta2.reshape(-1,1)))  #展开theta    
    
    start = time.time()
    result = optimize.fmin_cg(nnCostFunction, initial_nn_params, fprime=nnGradient, args=(input_layer_size,hidden_layer_size,out_put_layer,X_train,y_train,Lambda),maxiter=max_iter)
    print (u'执行时间：',time.time()-start)
    
    '''Theta1，theta2'''
    length = result.shape[0]
    Theta1 = result[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = result[hidden_layer_size*(input_layer_size+1):length].reshape(out_put_layer,hidden_layer_size+1)    

    '''预测'''
    p = predict(Theta1,Theta2,X_test)
    acuuracy = np.mean(np.float64(p == y_test.reshape(-1,1)))
    print (u"预测准确度为：%.4f%%"%(acuuracy*100))    
    
    return acuuracy

# 代价函数
def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
    length = nn_params.shape[0] # theta的中长度
    # 还原theta1和theta2
    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1)
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):length].reshape(num_labels,hidden_layer_size+1)
    
    m = X.shape[0]
    class_y = np.zeros((m,num_labels))      # 数据的y对应0-2，需要映射为0/1的关系
    # 映射y
    for i in range(num_labels):
        class_y[:,i] = np.int32(y==i).reshape(1,-1) # 注意reshape(1,-1)才可以赋值
     
    '''去掉theta1和theta2的第一列，因为正则化时从1开始'''    
    Theta1_colCount = Theta1.shape[1]    
    Theta1_x = Theta1[:,1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]    
    Theta2_x = Theta2[:,1:Theta2_colCount]
    # 正则化向theta^2
    term = np.dot(np.transpose(np.vstack((Theta1_x.reshape(-1,1),Theta2_x.reshape(-1,1)))),np.vstack((Theta1_x.reshape(-1,1),Theta2_x.reshape(-1,1))))
    
    '''正向传播,每次需要补上一列1的偏置bias'''
    a1 = np.hstack((np.ones((m,1)),X))      
    z2 = np.dot(a1,np.transpose(Theta1))    
    a2 = sigmoid(z2)  
    a2 = np.hstack((np.ones((m,1)),a2))
    z3 = np.dot(a2,np.transpose(Theta2))
    h  = sigmoid(z3)    
    '''代价'''
    J = -(np.dot(np.transpose(class_y.reshape(-1,1)),np.log(h.reshape(-1,1)))+
          np.dot(np.transpose(1-class_y.reshape(-1,1)),np.log(1-h.reshape(-1,1)))
          -Lambda*term/2)/m      

    return np.ravel(J)

# 梯度
def nnGradient(nn_params,input_layer_size,hidden_layer_size,num_labels,X,y,Lambda):
    length = nn_params.shape[0]
    Theta1 = nn_params[0:hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size,input_layer_size+1).copy()   # 这里使用copy函数，否则下面修改Theta的值，nn_params也会一起修改
    Theta2 = nn_params[hidden_layer_size*(input_layer_size+1):length].reshape(num_labels,hidden_layer_size+1).copy()
    m = X.shape[0]
    class_y = np.zeros((m,num_labels))      # 数据的y对应0-2，需要映射为0/1的关系    
    # 映射y
    for i in range(num_labels):
        class_y[:,i] = np.int32(y==i).reshape(1,-1) # 注意reshape(1,-1)才可以赋值
     
    '''去掉theta1和theta2的第一列，因为正则化时从1开始'''
    Theta2_colCount = Theta2.shape[1]    
    Theta2_x = Theta2[:,1:Theta2_colCount]
    
    Theta1_grad = np.zeros((Theta1.shape))  #第一层到第二层的权重
    Theta2_grad = np.zeros((Theta2.shape))  #第二层到第三层的权重
      
   
    '''正向传播，每次需要补上一列1的偏置bias'''
    a1 = np.hstack((np.ones((m,1)),X))
    z2 = np.dot(a1,np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m,1)),a2))
    z3 = np.dot(a2,np.transpose(Theta2))
    h  = sigmoid(z3)
    
    
    '''反向传播，delta为误差，'''
    delta3 = np.zeros((m,num_labels))
    delta2 = np.zeros((m,hidden_layer_size))
    for i in range(m):
        #delta3[i,:] = (h[i,:]-class_y[i,:])*sigmoidGradient(z3[i,:])  # 均方误差的误差率
        delta3[i,:] = h[i,:]-class_y[i,:]                              # 交叉熵误差率
        Theta2_grad = Theta2_grad+np.dot(np.transpose(delta3[i,:].reshape(1,-1)),a2[i,:].reshape(1,-1))
        delta2[i,:] = np.dot(delta3[i,:].reshape(1,-1),Theta2_x)*sigmoidGradient(z2[i,:])
        Theta1_grad = Theta1_grad+np.dot(np.transpose(delta2[i,:].reshape(1,-1)),a1[i,:].reshape(1,-1))
    
    Theta1[:,0] = 0
    Theta2[:,0] = 0          
    '''梯度'''
    grad = (np.vstack((Theta1_grad.reshape(-1,1),Theta2_grad.reshape(-1,1)))+Lambda*np.vstack((Theta1.reshape(-1,1),Theta2.reshape(-1,1))))/m
    return np.ravel(grad)

# S型函数    
def sigmoid(z):
    h = np.zeros((len(z),1))    # 初始化，与z的长度一致
    
    h = 1.0/(1.0+np.exp(-z))
    return h

# S型函数导数
def sigmoidGradient(z):
    g = sigmoid(z)*(1-sigmoid(z))
    return g

# 随机初始化权重theta
def randInitializeWeights(L_in,L_out):
    W = np.zeros((L_out,1+L_in))    # 对应theta的权重
    epsilon_init = (6.0/(L_out+L_in))**0.5
    W = np.random.rand(L_out,1+L_in)*2*epsilon_init-epsilon_init # np.random.rand(L_out,1+L_in)产生L_out*(1+L_in)大小的随机矩阵
    return W

# 预测
def predict(Theta1,Theta2,X):
    m = X.shape[0]

    '''正向传播，预测结果'''
    X = np.hstack((np.ones((m,1)),X))
    h1 = sigmoid(np.dot(X,np.transpose(Theta1)))
    h1 = np.hstack((np.ones((m,1)),h1))
    h2 = sigmoid(np.dot(h1,np.transpose(Theta2)))
    
    '''
    返回h中每一行最大值所在的列号
    - np.max(h, axis=1)返回h中每一行的最大值（是某个数字的最大概率）
    - 最后where找到的最大概率所在的列号（列号即是对应的数字）
    '''
 
    p = np.array(np.where(h2[0,:] == np.max(h2, axis=1)[0]))  
    for i in np.arange(1, m):
        t = np.array(np.where(h2[i,:] == np.max(h2, axis=1)[i]))
        p = np.vstack((p,t))
    return p    

def test_iter():
    max_iter_list = [60,70,80,90,100]
    accuracy_list = []
    for iters in max_iter_list:
        accuracy_list.append(neuralNetwork(4,10,3,iters))
   
    plt.rcParams['font.sans-serif'] = ['SimHei']        #解决中文字体
    plt.plot(max_iter_list, accuracy_list, color='r')
    plt.scatter(max_iter_list, accuracy_list,color='',marker='o', edgecolors='g', s=200)                         
    plt.title(u'准确率与迭代次数的关系',fontsize=20)
    plt.xlabel(u'迭代次数',fontsize=15)
    plt.ylabel(u'准确率',fontsize=15)
    plt.show()

if __name__ == "__main__":
    res = []
    for i in range(10):
        res.append(neuralNetwork(4,10,3,100))
    print(np.mean(res))    
    