#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:59:33 2019

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
import matplotlib.pyplot as plt
import os
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import scale
from pandas import DataFrame
from collections import Counter
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras import backend as K




aid = "AID_485314" ## change AID

X_trainPath = '/Volumes/selcuk/DeepDrug/dataset/resample/'+aid+'/X_train_raw.txt'
y_trainPath = '/Volumes/selcuk/DeepDrug/dataset/resample/'+aid+'/y_train_raw.txt'


X_valPath = '/Volumes/selcuk/DeepDrug/dataset/resample/'+aid+'/X_val_raw.txt'
y_valPath = '/Volumes/selcuk/DeepDrug/dataset/resample/'+aid+'/y_val_raw.txt'

X_testPath = '/Volumes/selcuk/DeepDrug/dataset/resample/'+aid+'/X_test_raw.txt'
y_testPath = '/Volumes/selcuk/DeepDrug/dataset/resample/'+aid+'/y_test_raw.txt'


X_train = pd.read_csv(X_trainPath, delimiter=",")
y_train = pd.read_csv(y_trainPath, delimiter=",")
X_val = pd.read_csv(X_valPath, delimiter=",")
y_val = pd.read_csv(y_valPath, delimiter=",")
X_test = pd.read_csv(X_testPath, delimiter=",")
y_test = pd.read_csv(y_testPath, delimiter=",")


X_val = np.array(X_val)
y_val = np.array(y_val)

X_test=np.array(X_test)
y_test=np.array(y_test)


rus = RandomUnderSampler(random_state=123)
X_resample,y_resample=rus.fit_sample(X_train,y_train[:,0]) # performing RUS 


X_resample=pd.DataFrame(X_resample)
y_resample=pd.DataFrame(y_resample)

X_train = np.array(X_resample) #balanced train dataset
y_train=np.array(y_resample)  #balanced train classes

X_train.shape


from collections import Counter
counter = Counter(y_train[:,0])
print(counter)  


def f1_score_threshold(threshold=0.5):
    def f1_score(y_true, y_predict):
        threshold_value = threshold
        y_predict = tf.cast(tf.greater(tf.clip_by_value(y_predict, 0, 1), threshold_value), tf.float32)
        true_positives = tf.round(tf.reduce_sum(tf.clip_by_value(y_true * y_predict, 0, 1)))
        predicted_positives = tf.reduce_sum(y_predict)
        precision_ratio = true_positives / (predicted_positives + 10e-6)

        possible_positives = tf.reduce_sum(tf.clip_by_value(y_true, 0, 1))
        recall_ratio = true_positives / (possible_positives + 10e-6)

        return (2 * recall_ratio * precision_ratio) / (recall_ratio + precision_ratio + 10e-6)

    return f1_score



model = Sequential([
     #Input Layer
     Dense(units=2000, input_dim=X_train.shape[1], activation='relu',kernel_initializer='glorot_uniform'),
     BatchNormalization(),
     Dropout(0.75),
     
     #First Hidden Layer
     Dense(units=500,activation='relu',kernel_initializer='glorot_uniform'),
     BatchNormalization(),
     Dropout(0.75), 
     
     
     #Second Hidden Layer
     Dense(units=10,activation='relu',kernel_initializer='glorot_uniform'),
     BatchNormalization(),
     Dropout(0.75),

     #Output Layer
     Dense(1,activation='sigmoid')      
])

model.summary()

import tempfile

initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')

model.save_weights(initial_weights)

optimizer = Adam(lr=0.0001)

model.compile(optimizer=optimizer, loss='binary_crossentropy',metrics=[f1_score_threshold(0.5), 'acc'])

model.load_weights(initial_weights)

history = model.fit(X_train,y_train, batch_size=64, epochs=60, validation_data = (X_val, y_val))



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
#pathFig = '/Volumes/selcuk/DeepDrug/Classification Results/'+aid+'/rus.png'

#plt.savefig(pathFig)



####### TRAIN RESULTS
y_pred=model.predict(X_train)
y_expected=pd.DataFrame(y_train)

pred = [1 if i >0.09 else 0 for i in y_pred]
Y_train = y_expected.values[:,0]

columns = ["True", "Prob"]
data = np.array([Y_train, y_pred[:,0]])
df = pd.DataFrame(data=data.T, columns=columns)


path = '/Volumes/selcuk/DeepDrug/Classification Results/'+aid+'/rus_train.txt'

df.to_csv(path, sep='\t', index=False)




####### VALIDATION RESULTS
y_pred=model.predict(X_val)
y_expected=pd.DataFrame(y_val)

pred = [1 if i >0.09 else 0 for i in y_pred]
Y_val = y_expected.values[:,0]

columns = ["True", "Prob"]
data = np.array([Y_val, y_pred[:,0]])
df = pd.DataFrame(data=data.T, columns=columns)


path = '/Volumes/selcuk/DeepDrug/Classification Results/'+aid+'/rus_val.txt'

df.to_csv(path, sep='\t', index=False)





####### TEST RESULTS
y_pred=model.predict(X_test)
y_expected=pd.DataFrame(y_test)

pred = [1 if i >0.09 else 0 for i in y_pred]
Y_test = y_expected.values[:,0]


columns = ["True", "Prob"]
data = np.array([Y_test, y_pred[:,0]])
df = pd.DataFrame(data=data.T, columns=columns)


path = '/Volumes/selcuk/DeepDrug/Classification Results/'+aid+'/rus_test.txt'

df.to_csv(path, sep='\t', index=False)


