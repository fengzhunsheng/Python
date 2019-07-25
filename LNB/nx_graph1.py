# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:14:38 2019

@author: 冯准生
"""

import networkx as nx
import matplotlib.pyplot as plt
G=nx.Graph()       #建立一个空的无向图
G.add_node(1)      #添加一个节点1
G.add_edge(2,3)    #添加一条边2-3（隐含添加节点2，3）
G.add_edge(3,2)    #无向图里2-3，3-2是同一条边
G.add_nodes_from([7,4,5,6])      #加点集合
G.add_cycle([4,5,6])             #加环
G.add_edges_from([(1,3),(1,7)])  #加边集合
print("nodes:",G.nodes())                       #点集合
print("edges:",G.edges())                       #边集合
print("number of edges:",G.number_of_edges())   #点的数目
print("number of nodes:",G.number_of_nodes())   #边的数目
degree_sequence = sorted([[n,d] for n, d in G.degree()],
                          key=lambda x:x[1], reverse=True) #结点的度
adj_nodes=[]
print("(n,degree):",degree_sequence)
#n是顶点，nbrs是顶点n的相邻顶点，是一个字典结构
for n,nbrs in G.adjacency():
    adj_node_n=[]
    #nbr表示跟n连接的顶点，attr表示这两个点连边的属性集合
    for nbr,attr in nbrs.items():
        adj_node_n.append(nbr)
    adj_nodes.append([n,adj_node_n])
print("adjacency of n:
    ",adj_nodes)

nx.draw_networkx(G,with_lables=True,node_size=400,node_color='b',edge_color='r')
plt.show()
#- node_size: 指定节点的尺寸大小(默认是300)
#- node_color: 指定节点的颜色 (默认是红色，例如'r'为红色,具体可查看手册)
#- node_shape: 节点的形状（默认是圆形，用字符串'o'标识，具体可查看手册）
#- alpha: 透明度 (默认是1.0，不透明，0为完全透明)
#- width: 边的宽度 (默认为1.0)
#- edge_color: 边的颜色(默认为黑色)
#- style: 边的样式(默认为实现，可选： solid|dashed|dotted,dashdot)
#- with_labels: 节点是否带标签（默认为True）
#- font_size: 节点标签字体大小 (默认为12)
#- font_color: 节点标签字体颜色（默认为黑色）
