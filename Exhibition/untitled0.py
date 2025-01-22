# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 12:04:56 2019

@author: Elton
"""

N=input()
arr = list(map(int, input().rstrip().split()))
win=max(arr)
counter=arr.count(win)
print(counter)