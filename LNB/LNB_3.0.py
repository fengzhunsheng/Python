# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:12:36 2019

@author: 冯准生
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd    #pandas,强大的数据处理能力
import random
#处理数据简洁，两行代码搞定
sourse_data=pd.read_csv("usair.txt",sep=' ')  #Dataframe
list=sourse_data.values.tolist()
#生成开始原图
G=nx.Graph()
G.add_edges_from(list)
############# 处理图 #############
train_set=[]
test_set=[]
for edge in list:
    if random.randint(0,100)<10:
        test_set.append(edge)
    else:
        train_set.append(edge)
##################################
#生成训练集图
G_train=nx.Graph()
G_train.add_edges_from(train_set)
#图的属性输出
degree_sequence = sorted([d for n, d in G_train.degree()], reverse=True)
print("nodes:",G_train.number_of_nodes())
print("edges:",G_train.number_of_edges(),end='\n')
print("Degree sequence", degree_sequence)
#画图
plt.figure(figsize=[90,60])  #matplotlib画布的宽和高
pos=nx.spring_layout(G)      #布局
#分别画点、边、结点标签
nx.draw_networkx_nodes(G_train, pos, node_size=400,node_color='g')
nx.draw_networkx_edges(G_train, pos, alpha=0.4)
nx.draw_networkx_labels(G_train,pos,lables=G.nodes,font_color='r')

plt.show()        
    