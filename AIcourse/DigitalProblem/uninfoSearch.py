# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 23:11:45 2019
@file:
@brief:

@author: 冯准生
@email:1565853379@qq.com
"""

import copy
import datetime

OPEN = []
CLOSED = []
NODE_NUM = 0
direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]

class State():
    def __init__(self, depth=0, state=None, par=None):
        self.depth = depth
        self.state = state
        self.par = par

def generate_child(cur_node,algType):
    num = len(cur_node.state)
    for i in range(0, num):
        for j in range(0, num):
            if cur_node.state[i][j] != 0:       # 找到0的位置
                continue
            for d in direction:                 # 四个偏移方向
                x = i + d[0]
                y = j + d[1]
                if x < 0 or x >= num or y < 0 or y >= num:          # 越界了
                    continue
                state = copy.deepcopy(cur_node.state)               # 复制父节点的状态
                state[i][j], state[x][y] = state[x][y], state[i][j] # 交换位置
                depth = cur_node.depth + 1
                child_node = State(depth,state,cur_node)
                
                if state in CLOSED:
                    continue
                
                global NODE_NUM
                NODE_NUM +=1
                        
                if algType == 0:            
                    OPEN.insert(0,child_node)
                else:
                    OPEN.append(child_node)

def print_path(node):
    '''
    输出路径
    :param node: 最终的节点
    :return: 总共移动地步数
    '''
    num = node.depth

    def show_START(START):
        print("---------------")
        for b in START:
            print(b)

    stack = []  
    while node.par is not None:
        stack.append(node.state)
        node = node.par
    stack.append(node.state)
    while len(stack) != 0:
        t = stack.pop()
        show_START(t)
    return num

def deep_prior(start,end):
    
    if start == end:
        print("start == end")
        return 0
    
    root = State(0,start)
    
    OPEN.append(root)
    while len(OPEN) != 0:
        top = OPEN.pop(0)
        CLOSED.append(top.state)
        if top.state == end:
            return print_path(top)
        
        if top.depth > 5:
            continue
        generate_child(top,0)
    
    print("No road !")         
    return -1

def wide_prior(start,end):
    root = State(0,start)

    OPEN.append(root)
    while len(OPEN) != 0:
        top = OPEN.pop(0)
        CLOSED.append(top.state)
        if top.state == end:
            return print_path(top)
        
        generate_child(top,1)
    
    print("No road !")        
    return -1

if __name__=='__main__':
    START=[[2,8,3],[1,0,4],[7,6,5]]
    GOAL=[[1,2,3],[8,0,4],[7,6,5]]
    
    start_t = datetime.datetime.now()
    length = wide_prior(START,GOAL)
    end_t = datetime.datetime.now()
    if length != -1:
        print("length =", length)
        print("time =", (end_t - start_t).total_seconds(), "s")
        print("nodes =",NODE_NUM)
        print("********************")
