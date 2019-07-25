# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:19:49 2019

@file: LocalNaiveBayes.py
@brief: this program use naive bayes to solve link predict problem.
    
@author: 冯准生
@email: 1565853379@qq.com
"""

import networkx as nx
import pandas as pd    #pandas,强大的数据处理能力
import community
import random
import math

########################   处理源数据   ###########################

#处理数据简洁，两行代码搞定
sourse_data=pd.read_csv("netscience.txt",sep=' ')  #Dataframe
edge_source=sourse_data.values.tolist()
#生成开始原图
G1=nx.Graph()
G1.add_edges_from(edge_source)
#图社区划分
part=community.best_partition(G1)
mod=community.modularity(part,G1)
#在part里面按照value的值来分开key
new_part={}
for k,v in part.items():
    new_part.setdefault(v,[]).append(k)
#根据new_part生成新的边
vertex_part=[]
for k,v in new_part.items():
    vertex_part.append(v)

edge_part=[]
for i in range(len(vertex_part)):
    edge_temp=[]
    for edge in edge_source:
        if edge[0] in vertex_part[i] and edge[1] in vertex_part[i]:
            edge_temp.append(edge)
    edge_part.append(edge_temp)


########################   LNB算法   ################################
vertex_set=G1.nodes
edge_set=edge_source
#print("vertex_set: ",len(vertex_set),"\n",vertex_set)
#print("edge_set:",len(edge_set),"\n",edge_set)
################   常量定义和集合划分    #####################

#定义要用的常量 
V=len(vertex_set)           #顶点数
E=len(edge_set)             #边的数目
TOTAL_TEST_NUM=(V-1)*V-0.9*E    #除了训练集外总的测试集的边的数目

TIMES_DEVISION=50   #对样本集作出的划分次数
TIMES_AUC=1000      #计算AUC所比较的次数

set_scores=[]   #保存节点对获得的分数

#测试集和训练集
train_set=[]
test_set=[]
non_existent_set=[]
G=nx.Graph()
non_observed_set=[]

##################   函数定义   ##########################
    
def set_devision():
    global test_set
    global train_set
    global non_existent_set
    global non_observed_set
    test_set.clear()
    train_set.clear()
    non_existent_set.clear()
    non_observed_set.clear()
    for edge in edge_set:
        if random.randint(0,100)<10:
            test_set.append(edge)
        else:
            train_set.append(edge)

    for v1 in vertex_set:
        for v2 in vertex_set:
            #所给样本边第一个顶点的值大于第二个
            if v1>v2 and [v1,v2] not in edge_set:   
                non_existent_set.append([v1,v2])
    non_observed_set=test_set[:]
    non_observed_set.extend(non_existent_set)
    #print("test_set: ",len(test_set),"\n",test_set)
    #print("train_set: ",len(train_set),"\n",train_set)
    #print("non_existent_set: ",len(non_existent_set),"\n",non_existent_set)
    #print("non_observed_set: ",len(non_observed_set),"\n",non_observed_set)

#计算AUC
def get_AUC():
    n1=n2=0
    for k in range(TIMES_AUC):
        #在missing link中随机选取一条边
        i=random.randint(0,len(test_set)-1)
        n1_score=set_scores[i]
        #在non_existent link中随机选取一条边
        j=random.randint(0,len(non_existent_set)-1)
        n2_score=set_scores[j+len(test_set)]
        if n1_score>n2_score:
            n1+=1
        elif n1_score==n2_score:
            n2+=1
    #根据n1,n2来计算AUC        
    AUC_scores=(n1+0.5*n2)/TIMES_AUC
    return AUC_scores

#计算precision,选取前（测试集个数）的分数
def get_precision():
    set_scores_sort=sorted(set_scores)
    set_scores_sort.sort(reverse=True)
    test_scores=set_scores[0:len(test_set)]
    num_top=0
    for score in set_scores_sort[0:len(test_set)]:
        if score in test_scores:
            num_top+=1
            
    return num_top/len(test_set)
        
#计算常数s
def get_s():
    M=(V-1)*V/2
    MT=len(train_set)
    s=M/MT-1
    return s
       
#获取x,y共同邻居
def get_Oxy(v1,v2):
    CN=[]
    for v in vertex_set:
        if G.has_edge(v1,v) and G.has_edge(v2,v) and v!=v1 and v!=v2:
            CN.append(v)
    return CN
    
#计算Rw
def get_Rw(w):
    k=G.degree(w)
    Nw=get_Nw(w)
    no_Nw=k*(k-1)/2-Nw
    return (Nw+1)/(no_Nw+1)
    
#计算顶点w邻居之间的连边
def get_Nw(w):
    cn=G.neighbors(w)
    num_pairs=0
    for v1 in cn:
        for v2 in cn:
           if G.has_edge(v1,v2):
               num_pairs+=1
    return num_pairs
    
def get_func(func,*args,**kwargs):
    func()
        
def r_LNB_CN():
    global set_scores
    set_scores.clear()
    for edge in non_observed_set:
        x=edge[0]
        y=edge[1]
        CN=get_Oxy(x,y)
        r=len(CN)*math.log(get_s())
        for w in CN:
            Rw=get_Rw(w)
            r+=math.log(Rw)
        set_scores.append(r)

  
if __name__=='__main__':
    AUC_scores=0
    precision_scores=0
    for i in range(TIMES_DEVISION):
        set_devision()
        G.clear()
        G.add_edges_from(train_set)  #生成训练集的图
        get_func(r_LNB_CN)
        score_1=get_AUC()
        AUC_scores+=score_1
        score_2=get_precision()
        precision_scores+=score_2
    AUC_scores/=TIMES_DEVISION
    precision_scores/=TIMES_DEVISION
    print(AUC_scores)
    print(precision_scores)