# -*- coding: utf-8 -*-
"""

@author: Elton
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

#loading the dataset
dataset= pd.read_csv("memes.csv",header=None)
dataset.fillna(value="None",inplace=True)

#reformatting the data
personal = dataset.iloc[1:,:10]
y=10
a=np.array([])
xmode=np.empty([1,2])
#columns to rows
obs=[[]for i in range (16)]
for i in range (16):
    dark=np.array([])
    dank=np.array([])

#    print(dod)
    temp= dataset.iloc[1:,y:y+4]
    y+=4

   
    demp=np.array(temp.iloc[:,2:4].values.astype(int))
#    print (demp)
    m= stats.mode(demp)
#    print(m[0])
    remp=np.array([np.mean(temp.iloc[:,2].astype(int))])
    nemp=np.array([np.mean(temp.iloc[:,3].astype(int))])
    n=m[0]
    for r in range(71):
        dark=np.append(dark,values=remp , axis=0)
        dank=np.append(dank,values=nemp,axis=0)
    for r in range(70):
        xmode=np.append(xmode,values=m[0],axis=0)
    ls =[personal,temp,pd.DataFrame(dark).iloc[1:,:],pd.DataFrame(dank).iloc[1:,:]]
    
    obs[i] = pd.concat(ls, axis=1, ignore_index=True)

datasheet= pd.concat(obs)
#adjusting the observations

xmode= xmode[1:,:]#dank, dark modes
#dependent and independent
x=datasheet.iloc[:, [2,3,4]].values
y=datasheet.iloc[:,11:14].values #relat,dank, dark og
xmeans=datasheet.iloc[:,14:16].values #dank,dark means
#one hot encoding a complex column
arr=[]
for i in x[:,2]:
    if "Marvel" in i :
        arr.append(1)
    else:
        arr.append(0)
datasheet['Marvel'] = arr 
arr=[]
for i in x[:,2]:
    if "Harry" in i :
        arr.append(1)
    else:
        arr.append(0)
datasheet['Harry'] = arr    
arr=[]   
for i in x[:,2]:
    if "Star" in i :
        arr.append(1)
    else:
        arr.append(0)
datasheet['Star'] = arr  

#label encoding and one hot encoding the rest of the data
#personal= sex, marvel, harry,starwars,friends,football,tweet
personal=datasheet.iloc[:,[3,-1,-2,-3,6,7]] 
age=datasheet.iloc[:,2:3].values.astype(int)
stream=datasheet.iloc[:,8:9]
freq=datasheet.iloc[:,9:10]
typeofmeme=datasheet.iloc[:,10:11]
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
personal=personal.apply(LabelEncoder().fit_transform).values
stream=stream.apply(LabelEncoder().fit_transform)
freq= freq.apply(LabelEncoder().fit_transform)
typeofmeme=typeofmeme.apply(LabelEncoder().fit_transform)

onehotencoder = OneHotEncoder()
stream= onehotencoder.fit_transform(stream).toarray()
freq= onehotencoder.fit_transform(freq).toarray()
typeofmeme= onehotencoder.fit_transform(typeofmeme).toarray()



#person tags to meme tags 
#0: Harry Potter
#1: Marvel
#2: Star Wars
#3: Tweet
#4: football
#5: general
common=np.zeros([1120,1])
for i in range(1120):
    if datasheet.iloc[i,-1]==1 and typeofmeme[i,2]==1:
        common[i]=1
   
        
    if datasheet.iloc[i,-2]==1 and typeofmeme[i,0]==1:
        common[i]=1
   
    
    if datasheet.iloc[i,-3]==1 and typeofmeme[i,1]==1:
        common[i]=1
    

    if typeofmeme[i,5]==1:
        common[i]=1
    
    
    if personal[i,-2]==1 and typeofmeme[i,4]==1:
        common[i]=1
    

    if personal[i,-1]==1 and typeofmeme[i,3]==1:
        common[i]=1
    
#avoiding the dummy variable trap

typeofmeme=typeofmeme[:,1:]
stream=stream[:,1:]
freq=freq[:,1:]


#personalnew= age, sex, marvel, harry,starwars,friends,football,tweet,stream(endcode) ,freq(encode)
personalnew=pd.concat([pd.DataFrame(age),pd.DataFrame(personal), pd.DataFrame(stream), pd.DataFrame(freq)], axis=1, ignore_index=True    )
#personalnew=pd.concat([pd.DataFrame(age),pd.DataFrame(personal), pd.DataFrame(freq)], axis=1, ignore_index=True    )


#\\\\\\\\meme_data= type, mean danrk,mode danrk,wht they felt\\\\\\\\\\\\\ 

#meme_data=pd.concat([pd.DataFrame(typeofmeme), pd.DataFrame(xmeans), pd.DataFrame(xmode), pd.DataFrame(y[:,1:3].astype(int))],axis=1,ignore_index=True)
meme_data=pd.concat([pd.DataFrame(typeofmeme), pd.DataFrame(xmeans), pd.DataFrame(xmode)],axis=1,ignore_index=True)




#x is our Independent variables
x=pd.concat([pd.DataFrame(common),pd.DataFrame(personalnew),pd.DataFrame(meme_data)], axis=1, ignore_index=True)



#modifying the Dependent variables
ynext= np.empty(shape=(1120,1))
for i in range(1120):
    if int(y[i,0])>2:
        ynext[i,0]=1
    else:
        ynext[i,0]=0

yd=pd.DataFrame(y)



#importing and redying ann
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x,ynext, test_size = 0.1)
results = pd.DataFrame()


import keras
from keras.models import Sequential
from keras.layers import Dense

##initialisation of ann
#classifier= Sequential()
#
##adding layers
#classifier.add(Dense(output_dim=20, init = "uniform", activation = "relu", input_dim= 23))
#
#classifier.add(Dense(output_dim=20, init = "uniform", activation = "relu"))
#
#classifier.add(Dense(output_dim=1, init = "uniform", activation = "sigmoid"))
#
#classifier.compile(optimizer="adam", loss = "binary_crossentropy", metrics = ["accuracy"])
#
#classifier.fit(x_train, y_train, batch_size = 50, nb_epoch = 100)
#
#y_pred = classifier.predict(x_test)
#
#for p in range(len(y_pred)):
#    if y_pred[p]>0.6:
#        y_pred[p]=1
#    else:
#        y_pred[p]=0
#        

from sklearn.metrics import confusion_matrix
#cm=confusion_matrix(y_test,y_pred)
#print(cm)
#
#acc=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])*100
#print(acc)
#

for rep in [5,8,11,13,16,19,23,25,30,35,50]:
    for ep in [1,10,100,200,300,500]:
        for batch in [10,20,30,50]:
            classifier= Sequential()
            
#adding    layers
            classifier.add(Dense(output_dim=rep, init = "uniform", activation = "relu", input_dim= 23))
            
            classifier.add(Dense(output_dim=rep, init = "uniform", activation = "relu"))

            classifier.add(Dense(output_dim=1, init = "uniform", activation = "sigmoid"))
    
            classifier.compile(optimizer="adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    
            classifier.fit(x_train, y_train, batch_size = 50, nb_epoch = ep)
    
            y_pred = classifier.predict(x_test)
    
            for p in range(len(y_pred)):
                if y_pred[p]>0.6:
                    y_pred[p]=1
                else:
                    y_pred[p]=0
            cm=confusion_matrix(y_test,y_pred)
            print(cm)
            
            acc=(cm[0,0]+cm[1,1])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])*100
            print(acc)        
                    
            tempr={'cm':[cm],'acc':[acc],'No. of nodes':[rep],'No. of epoch':[ep],'batch size':[batch]}
            res=pd.DataFrame(data=tempr)
            results=results.append(res)
            print("batch",batch,"ep",ep,"rep",rep)