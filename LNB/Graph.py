# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:24:42 2019

@author: 冯准生
"""
    
import matplotlib.pyplot as plt
import networkx as nx

G = nx.karate_club_graph()
print("Node Degree")
for v in G:
     print('%s %s' % (v, G.degree(v)))

nx.draw_circular(G, with_labels=True)
plt.show()