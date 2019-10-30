# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 09:58:33 2019
@file:   digitalProblem.py
@brief:  this file use A* algorithm to solve 8-puzzle problem
         the h(n) is manhattan distance and falsePostion number
         the manhattan is more efficient than falsePostion number

@author: 冯准生
@email:1565853379@qq.com
"""

import heapq
import copy
import re
import datetime
import numpy as np

START = []  # 给定状态
GOAL = []  # 目标状态

# 4个方向
direction = [[0, 1], [0, -1], [1, 0], [-1, 0]]

# OPEN表
OPEN = []

# 扩展节点的总数
SUM_NODE_NUM = 0


# 状态节点
class State(object):
    def __init__(self, gn=0, hn=0, state=None, hash_value=None, par=None):
        '''
        初始化
        :param gn: gn是初始化到现在的距离
        :param hn: 启发距离
        :param state: 节点存储的状态
        :param hash_value: 哈希值，用于判重
        :param par: 父节点指针
        '''
        self.gn = gn
        self.hn = hn
        self.fn = self.gn + self.hn
        self.child = []                         # 孩子节点
        self.par = par                          # 父节点
        self.state = state                      # 局面状态
        self.hash_value = hash_value            # 哈希值

    def __lt__(self, other):    # 用于堆的比较，返回距离最小的
        return self.fn < other.fn

    def __eq__(self, other):    # 相等的判断
        return self.hash_value == other.hash_value

    def __ne__(self, other):    # 不等的判断
        return not self.__eq__(other)

def false_pos_dis(cur_state, end_state):
    dist = 0
    N = len(cur_state)
    for i in range(N):
        for j in range(N):
            if cur_state[i][j] != end_state[i][j]:
                dist +=1
    
    return dist

def manhattan_dis(cur_state, end_state):        
    '''
    计算曼哈顿距离
    :param cur_state: 当前状态
    :return: 到目的状态的曼哈顿距离之和
    '''
    dist = 0
    N = len(cur_state)                  
    m = np.matrix(end_state)    # 将list转化为matrix
    for i in range(N):
        for j in range(N):
            num = cur_state[i][j]
            if num == 0:
                continue
            else:
                pos = np.argwhere(m==num)       # 用此函数获取对应元素的下标
                x = pos[0][0]
                y = pos[0][1]
                dist += (abs(x-i)+abs(y-j))     #每个方向曼哈顿距离等于当前和目标下标之差，再相加

    return dist

def generate_child(cur_node, end_node, hash_set, open_table, dis_fn):
    '''
    生成子节点函数
    :param cur_node:   当前节点
    :param end_node:   最终状态节点
    :param hash_set:   哈希表，用于判重
    :param open_table: OPEN表
    :param dis_fn:     距离函数
    :return: None
    '''
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
                h = hash(str(state))            # 哈希时要先转换成字符串
                if h in hash_set:               # 重复了
                    continue
                hash_set.add(h)                 # 加入哈希表
                gn = cur_node.gn + 1            # 已经走的距离函数
                hn = dis_fn(state, end_node.state)                  # 启发的距离函数
                node = State(gn, hn, state, h, cur_node)            # 新建节点
                cur_node.child.append(node)     # 加入到孩子队列
                heapq.heappush(open_table, node)                    # 加入到堆中
                
                global SUM_NODE_NUM             # 记录扩展节点的个数
                SUM_NODE_NUM += 1

def print_path(node):
    '''
    输出路径
    :param node: 最终的节点
    :return: 总共移动地步数
    '''
    num = node.gn

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


def A_start(start, end, distance_fn, generate_child_fn):
    '''
    A*算法
    :param start: 起始状态
    :param end: 终止状态
    :param distance_fn: 距离函数，可以使用自定义的
    :param generate_child_fn: 产生孩子节点的函数
    :return: 如果没有路径，返回-1；否则返回移动地步数
    '''
    root = State(0, 0, start, hash(str(START)), None)               # 根节点
    end_state = State(0, 0, end, hash(str(GOAL)), None)             # 最后的节点
    if root == end_state:
        print("start == end !")
        return 0

    OPEN.append(root)
    heapq.heapify(OPEN)         # 将OPEN列表转化为小顶堆

    node_hash_set = set()       # 存储节点的哈希值
    node_hash_set.add(root.hash_value)

    while len(OPEN) != 0:
        top = heapq.heappop(OPEN)               # 在OPEN堆中取出第一个，即最小值
        if top == end_state:    # 结束后直接输出路径
            return print_path(top)
        # 产生孩子节点，孩子节点加入OPEN表
        generate_child_fn(cur_node=top, end_node=end_state, hash_set=node_hash_set,
                          open_table=OPEN, dis_fn=distance_fn)

    print("No road !")          # 没有路径
    return -1


def test_from_file():
    def read_START(START, line, N):
        pattern = re.compile(r'\d+')  # 正则表达式提取数据
        res = re.findall(pattern, line)
        t = 0
        tmp = []
        for i in res:
            t += 1
            tmp.append(int(i))
            if t == N:
                t = 0
                START.append(tmp)
                tmp = []
            
          
    f = open("./infile.txt")
    NUMBER = int(f.readline()[-2])  
    n = 1
    for i in range(NUMBER):
        l = []
        for j in range(NUMBER):
            l.append(n)
            n += 1
        GOAL.append(l)
    GOAL[NUMBER - 1][NUMBER - 1] = 0

    for line in f:  # 读取每一行数据
        global OPEN
        global SUM_NODE_NUM
        OPEN = []   # 这里别忘了清空
        START = []
        read_START(START, line, NUMBER)
        SUM_NODE_NUM = 0
        start_t = datetime.datetime.now()
        # 这里添加5秒超时处理，可以根据实际情况选择启发函数
        length = A_start(START, GOAL, manhattan_dis, generate_child, time_limit=10)
        end_t = datetime.datetime.now()
        if length != -1:
            print("length =", length)
            print("time = ", (end_t - start_t).total_seconds(), "s")
            print("Nodes =", SUM_NODE_NUM)

def test_from_io():
    global GOAL
    global START
    START=[[2,8,3],[1,0,4],[7,6,5]]
    GOAL=[[1,2,3],[8,0,4],[7,6,5]]
    
    start_t = datetime.datetime.now()
    length = A_start(START, GOAL, manhattan_dis, generate_child)
    end_t = datetime.datetime.now()
    if length != -1:
        print("length =", length)
        print("time = ", (end_t - start_t).total_seconds(), "s")
        print("Nodes =", SUM_NODE_NUM)
        print("********************")

if __name__ == '__main__':
    test_from_io()
    