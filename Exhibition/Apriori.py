# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 16:24:49 2019

@author: Elton
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from itertools import combinations, chain
import pyfpgrowth

#loading the dataset
dataset= pd.read_csv("memes.csv",header=None)
dataset.fillna(value="None",inplace=True)


x = dataset.iloc[1:,:10]
y=10
v=0

users=[[]for o in range(16)]
#columns to rows

for i in range (16):

    liked=[]
#    print(disc)
    temp= dataset.iloc[1:20,y:y+4]
    y+=4     
#    print(m[0])

    #liked disliked
    e=pd.to_numeric(temp.iloc[:,1])
    for b in range(1,len(e)+1):
        if e[b]>2:
            e[b]=1
            liked.append(str(dataset.iloc[b,1])[:4])
        
           
        else:
            e[b]=0
            
    users[v].append(liked)
    v+=1        
   
          
    #dark or dank       
    

    
   

for i in range(16):
    users[i]=pd.DataFrame(users[i])
    
problem=pd.concat(users) 
ttransactions=[]
for i in range (16):    
    ttransactions.append([str(problem.values[i,j]) for j in range(19)])


from apyori import apriori
rules = apriori(ttransactions, min_support = 3, min_confidence = 0.2, min_lift = 3, min_length = 2)

# Visualising the results
results = list(rules)

import pyfpgrowth
patterns =  pyfpgrowth.find_frequent_patterns(ttransactions,2)
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)






