# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 17:23:31 2019
@file:  ID3.py
@brief: this is a work to solve the experience of AI.
        I use the ID3 algorithm to realize decision tree.

@author: 冯准生
@email:1565853379@qq.com
"""

import operator
from math import log
import pandas as pd
import Draw_decisionTree

#根据属性分割数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featvec in dataSet: 
        if featvec[axis] == value:          #每行中第axis个元素和value相等  
            #删除对应的元素，并将此行，加入到rerDataSet
            reducedFeatVec = featvec[:axis]
            reducedFeatVec.extend(featvec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#计算香农熵  #计算数据集的香农熵 == 计算数据集类标签的香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)               #数据集样本点个数
    labelCounts = {}                        #类标签
    for featVec in dataSet:                 #统计数据集类标签的个数，字典形式
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key]/numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

#根据香农熵，选择最优的划分方式    #根据某一属性划分后，类标签香农熵越低，效果越好
def chooseBestFeatureToSplit(dataSet):
    baseEntropy = calcShannonEnt(dataSet)   #计算数据集的香农熵
    numFeatures = len(dataSet[0])-1
    bestInfoGain = 0.0                      #最大信息增益
    bestFeature = 0                         #最优特征
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]      #所有子列表（每行）的第i个元素，组成一个新的列表
        uniqueVals = set(featList)          #去重
        newEntorpy = 0.0
        for value in uniqueVals:            #数据集根据第i个属性进行划分，计算划分后数据集的香农熵
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/len(dataSet)
            newEntorpy += prob*calcShannonEnt(subDataSet)
        infoGain = baseEntropy-newEntorpy   #划分后的数据集，香农熵越小越好，即信息增益越大越好
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

#如果数据集已经处理了所有属性，但叶子结点中类标签依然不是唯一的，此时需要决定如何定义该叶子结点。
#这种情况下，采用多数表决方法，对该叶子结点进行分类
def majorityCnt(classList):                 #传入参数：叶子结点中的类标签
    classCount = {}
    for vote in classList:                  #统计不同类别的标签的数目
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

#递归的创建树
def createTree(dataSet, labels):            #传入参数：数据集，属性标签（属性标签作用：在输出结果时，决策树的构建更加清晰）
    classList = [example[-1] for example in dataSet]        #数据集样本的类标签
    if classList.count(classList[0]) == len(classList):     #如果数据集样本属于同一类，说明该叶子结点划分完毕
        return classList[0]
    if len(dataSet[0]) == 1:                #如果数据集样本只有一列（该列是类标签），说明所有属性都划分完毕，则根据多数表决方法，对该叶子结点进行分类
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)            #根据香农熵，选择最优的划分方式
    bestFeatLabel = labels[bestFeat]        #记录该属性标签
    myTree = {bestFeatLabel:{}}             #树
    del(labels[bestFeat])                   #在属性标签中删除该属性
    #根据最优属性构建树
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        subDataSet = splitDataSet(dataSet, bestFeat, value) #新的子数据集
        myTree[bestFeatLabel][value] = createTree(subDataSet, subLabels)    #构建子树
    return myTree


#测试算法：使用决策树，对待分类样本进行分类
def classify(inputTree, featLabels, testVec):               #传入参数：决策树，属性标签，待分类样本
    firstStr = list(inputTree.keys())[0]    #树根代表的属性
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)  #树根代表的属性，所在属性标签中的位置，即第几个属性
    key = testVec[featIndex]                #取出测试向量中该属性对应的值
    if key not in secondDict.keys():        #有可能key不在该字典中
        return None
    if type(secondDict[key]).__name__ == 'dict':
        classLabel = classify(secondDict[key], featLabels, testVec)
    else:
        classLabel = secondDict[key]
    
    return classLabel

def load_data(filename:str):
    data_source = pd.read_csv(filename, sep='[ |\t]', engine='python')
    data_1 =data_source.values.tolist()
    data = []
    for item in data_1:
        if item[0] < 5.5:
            item[0] = 'SL_1'
        elif item[0] < 6.5:
            item[0] = 'SL_2'
        else:
            item[0] = 'SL_3'
            
        if item[1] < 3.5:
            item[1] = 'SW_1'
        else:
            item[1] = 'SW_2'
            
        if item[2] < 2:
            item[2] = 'PL_1'
        elif item[2] < 5:
            item[2] = 'PL_2'
        else:
            item[2] = 'PL_3'
            
        if item[3] < 1:
            item[3] = 'PW_1'
        elif item[3] < 1.8:
            item[3] = 'PW_2'
        else:
            item[3] = 'PW_3'
        
        data.append(item)
            
    return data
    
if __name__ == '__main__':
    #训练决策树
    data_train = load_data('traindata.txt')
    feat_labels =['SL','SW','PL','PW','FC']
    tree = createTree(data_train,labels=feat_labels[:])
    Draw_decisionTree.createPlot(tree)
    
    #根据决策树，判断正确率
    data_test = load_data('testdata.txt')
    numCorrect=0
    for sample in data_test:
        testvec=sample[0:4]
        if classify(inputTree=tree,featLabels=feat_labels,testVec=testvec)==sample[4]:
            numCorrect += 1
    print("correct rating = %.2f%%"%(numCorrect/len(data_test)*100))