#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 11:38:50 2019

@author: selcukkorkmaz
"""

from numpy.random import seed
seed(123)
from tensorflow import set_random_seed
set_random_seed(123)
import pandas as pd
import numpy as np
import keras
import seaborn as sns
import os
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
import collections, numpy
from pandas import DataFrame
import math
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras import backend as K



aid = "AID_485314" # change AID

X_trainPath = '/Volumes/selcuk/DeepDrug/dataset/resample/'+aid+'/RAW/X_train_raw.txt'
y_trainPath = '/Volumes/selcuk/DeepDrug/dataset/resample/'+aid+'/RAW/y_train_raw.txt'
X_testPath = '/Volumes/selcuk/DeepDrug/dataset/resample/'+aid+'/RAW/X_test_raw.txt'
y_testPath = '/Volumes/selcuk/DeepDrug/dataset/resample/'+aid+'/RAW/y_test_raw.txt'


X_train = pd.read_csv(X_trainPath, delimiter=",")
y_train = pd.read_csv(y_trainPath, delimiter=",")
X_test = pd.read_csv(X_testPath, delimiter=",")
y_test = pd.read_csv(y_testPath, delimiter=",")

X_train = np.array(X_train)
X_test=np.array(X_test)
y_train=np.array(y_train)
y_test=np.array(y_test)


X_resample,y_resample=SMOTE(random_state=123).fit_sample(X_train,y_train[:,0]) # performing SMOTE 

X_resample=pd.DataFrame(X_resample)
y_resample=pd.DataFrame(y_resample)

X_train = np.array(X_resample) #balanced train dataset
y_train=np.array(y_resample)  #balanced train classes

X_train.shape

#resamplePathX = '/Volumes/selcuk/resample/'+aid+'/SMOTE/X_resample_smote.txt'
#resamplePathY = '/Volumes/selcuk/resample/'+aid+'/SMOTE/y_resample_smote.txt'


#export_csv_X = X_resample.to_csv (resamplePathX, index = None, header=True) #write balanced dataset
#export_csv_y = y_resample.to_csv (resamplePathY, index = None, header=True) #write balanced dataset



def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


model = Sequential([
     #Input Layer
     Dense(units=1000, input_dim=X_train.shape[1], activation='relu',kernel_initializer='glorot_uniform'),
     BatchNormalization(),
     Dropout(0.5),
     
     #First Hidden Layer
     Dense(units=500,activation='relu',kernel_initializer='glorot_uniform'),
     BatchNormalization(),
     Dropout(0.5), 
     
     #Second Hidden Layer
     Dense(units=100,activation='relu',kernel_initializer='glorot_uniform'),
     BatchNormalization(),
     Dropout(0.5),

     #Output Layer
     Dense(1,activation='sigmoid')      
])

model.summary()

optimizer = Adam(lr=0.0001)

model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=['accuracy', f1_m])
history = model.fit(X_train,y_train, batch_size=256, epochs=10, validation_split = 0.10)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


y_pred=model.predict(X_test)
y_expected=pd.DataFrame(y_test)

pred = y_pred.round()[:,0]
Y_test = y_expected.values[:,0]


cnf_matrix=confusion_matrix(y_expected,y_pred.round())
cnf_matrix

tn = cnf_matrix[0,0]
tp = cnf_matrix[1,1]
fn = cnf_matrix[1,0]
fp = cnf_matrix[0,1]

bacc = ((tp/(tp+fn))+(tn/(tn+fp)))/2
pre = tp/(tp+fp)
rec = tp/(tp+fn)
f1 = 2*pre*rec/(pre+rec)
mcc = ((tp*tn) - (fp*fn))/math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
auc = roc_auc_score(Y_test, y_pred[:,0])


performance = {'BACC': [bacc], 'Precision': [pre], 'Recall': [rec], 'F1': [f1],
               'MCC': [mcc], 'AUC': [auc]}
performanceDF = pd.DataFrame(performance)
path = '/Volumes/selcuk/DeepDrug/Classification Results/'+aid+'/smote_perf.txt'
performanceDF.to_csv(path, sep='\t', index=False)


columns = ["True", "Pred", "Prob"]
data = np.array([Y_test, pred, y_pred[:,0]])
df = pd.DataFrame(data=data.T, columns=columns)


path = '/Volumes/selcuk/DeepDrug/Classification Results/'+aid+'/smote.txt'

df.to_csv(path, sep='\t', index=False)

