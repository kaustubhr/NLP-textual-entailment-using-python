# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:52:29 2020

@author: pk
"""
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report

from xml.dom import minidom

vocab_size = 0
a=0

def pathToLists(pathname):
    mydoc = minidom.parse(pathname)
    tTag = mydoc.getElementsByTagName('t')
    hTag = mydoc.getElementsByTagName('h')
    #lab = mydoc.getElementsById()
    lTag = mydoc.getElementsByTagName('pair')
    
    
    
    
    
    
    
    
    
    tTaglist = []
    hTaglist = []
    lTagList = []
    
    for elem in tTag:                
        tTaglist.append(elem.firstChild.data)
    for elem in hTag:
        hTaglist.append(elem.firstChild.data)
    for elem in lTag:
        lTagList.append(elem.getAttribute("value"))
    
    
    tTaglist = [i.split(" ") for i in tTaglist]
    hTaglist = [i.split(" ") for i in hTaglist]
    
    
    
    
    
    print('p = ')     
    print(tTaglist)
    print('h = ')
    print(hTaglist)
    print('l = ')
    print(lTagList)
    integerEncode(tTaglist,hTaglist,lTagList)
    
def integerEncode(tTaglist,hTaglist,lTagList):
    s=set()
    s2 = set() #for labels list
    
    #integer coding true false 
    for ele in lTagList:
        s2.add(ele)
    temp2 = {i: j for j, i in enumerate(s2,start=1)}
    print("temp2 is ")
    print(temp2)
    
    
    #adding tag words to a set
    for ele in tTaglist:
        for elem in ele:
            s.add(elem)
    
    
    #adding hypothesis words into the same set
    for ele in hTaglist:
        for elem in ele:
            s.add(elem)
    
    #vocab of the input files (includes tTaglist as well as hTaglist words)
    temp = {i: j for j, i in enumerate(s,start=1)}
    print("vocab length is")
    
    vocab_size = len(temp) + 1
    print(vocab_size)
    
    
    
    tnew=[]
   
   
    for ele in tTaglist:
        tnew1=[]
        count=0
        for elem in ele:
            tnew1.append(temp.get(elem))
            count=count+1
        #padding till max words in premise tag    
        while(count<61):
            tnew1.append(0)
            count=count+1
        tnew.append(tnew1)
        
    hnew=[]
    
    for ele in hTaglist:
        hnew1=[]
        count=0
        for elem in ele:
            hnew1.append(temp.get(elem))
            count=count+1
        #padding till max words in premise tag
        while(count<61):
            hnew1.append(0)
            count=count+1
        hnew.append(hnew1)
        
    lNew = []
    for ele in lTagList:
        lNew.append([temp2.get(ele)])
        
        
    
        
    
    
    
    
    print("tnew is")
    print(tnew)
    print("hnew is")
    print(hnew)
    print("lNew is")
    print(lNew)
    preparing(tnew,hnew,lNew,vocab_size)
    
    #tnew has list of tag integers , hnew has list of hypothesis integers, padding total length 61 
            
def preparing(tList,hList,labelList,vocab_size):
    tNP = np.asarray(tList, dtype=np.int32)
    hNP = np.asarray(hList, dtype=np.int32)
    lNP = np.asarray(labelList, dtype=np.int32) 
    #lNP = np.asarray(labelList, dtype=np.int32)
    #lNP = labelList
    print("t numpy is")
    print(tNP)
    print("h numpy array is")
    print(hNP)
    print("l is")
    print(lNP)
    print(type(tNP))
    print(type(hNP))
    print(type(lNP))
    
    thNP= np.concatenate((tNP,hNP),axis=1)
    print("t+h numpy is ")
    print(thNP)
    print("type of t+h numpy is")
    print(type(thNP))
    
    
    
    #model = Sequential()
    #print(model)
    print("shape of tnp is")
    print(tNP.shape)
    print("shape of hNP is ")
    print(hNP.shape)
    print("shape of lNP is ")
    print(lNP.shape)
    print("shape of t+h NP is ")
    print(thNP.shape)
    
   
    print("dtype of tnp is")
    print(tNP.dtype)
    print("dtype of hNP is ")
    print(hNP.dtype)
    print("dtype of lNP is ")
    print(lNP.dtype)
    print("vocab size is ")
    print(vocab_size)
    
    
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(vocab_size+1, 64))
    #model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM()))
    model.add(tf.keras.layers.Dense(8,activation='relu'))
    
    model.add(tf.keras.layers.Dense(1))
    
    
    
    #model.compile(loss=tf.keras.losses.categorical_crossentropy(from_logits=True),optimizer=tf.keras.optimizers.Adam(1e-4),metrics=['accuracy'])
    #model.compile(optimizer='sgd', loss='mean_squared_error')
    model.compile('sgd', loss='mse', metrics=[tf.keras.metrics.BinaryAccuracy()])
    model.summary()
    
    
    
    model.fit(thNP, lNP,  batch_size=50,epochs=5)
    result = model.evaluate(thNP,lNP)
    
    #result2 = model.predict(result)
    print("result is")
    print(result)
    
    """TP = tf.math.count_nonzero(lNP)
    print("size of lnp is")
    print(len(lNP))
    print("TP is")
    print(TP)
    """
    
    global a
    if(a==0):
        a=1
        print("give path to test")
        pathToLists(str(input()))
    
    if(a==1):
        sys.exit()
    
   
    
    
  

    

def main():
    pathToLists(sys.argv[1])

if __name__ == "__main__":
    main()
    
    