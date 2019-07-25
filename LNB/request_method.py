# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 22:00:20 2019

@author: 冯准生
"""

import requests
from bs4 import BeautifulSoup
import re
r=requests.get('https://book.douban.com/subject/1084336/comments/')
soup=BeautifulSoup(r.text,'lxml')
pattern=soup.find_all('span','short')
for item in pattern:
    print(item.string)
pattern_s=re.compile('<span class="user-starts allstar(.*?) rating"')
p=re.findall(pattern_s,r.text)
print(p)