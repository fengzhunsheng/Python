# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 16:35:32 2019

@brief: use pd to load data to list from txt file
        use nx and plt to draw figures 

@author: 冯准生
"""

import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd    #pandas,强大的数据处理能力
#处理数据简洁，两行代码搞定
sourse_data=pd.read_csv("usair.txt",sep=' ')  #Dataframe格式
list=sourse_data.values.tolist()
#生成图
G=nx.Graph()
G.add_edges_from(list)
#图的属性输出
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
print("nodes:",G.number_of_nodes())
print("edges:",G.number_of_edges(),end='\n')
print("Degree sequence", degree_sequence)
#画图
plt.figure(figsize=[120,80])  #matplotlib画布的宽和高
pos=nx.spring_layout(G)      #布局
#分别画点、边、结点标签
nx.draw_networkx_nodes(G, pos, node_size=400,node_color='g')
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_labels(G,pos,lables=G.nodes,font_color='r')

plt.show()