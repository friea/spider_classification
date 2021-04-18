# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:51:36 2021

@author: Friea
"""
# %%
#kütüphanelerin yüklenmesi
import numpy as np 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix
import warnings
import pickle
warnings.filterwarnings("ignore")
# %%
#Modelin oluşturulması
classifier=keras.Sequential()
classifier.add(tf.keras.layers.Conv2D(16,3,3, input_shape=(64,64,3),activation='tanh', padding="valid"))
classifier.add(tf.keras.layers.Conv2D(16,3,3,activation='tanh', padding="valid"))
classifier.add(BatchNormalization())
classifier.add(tf.keras.layers.MaxPool2D(pool_size=(1,1), padding="valid"))

classifier.add(tf.keras.layers.Conv2D(16,3,3,activation='tanh', padding="same"))
classifier.add(BatchNormalization())
classifier.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1), padding="same"))

classifier.add(tf.keras.layers.Conv2D(16,3,3,activation='tanh', padding="same"))
classifier.add(BatchNormalization())
classifier.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1), padding="same"))

classifier.add(tf.keras.layers.Conv2D(16,3,3,activation='tanh', padding="same"))
classifier.add(BatchNormalization())
classifier.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1), padding="same"))

classifier.add(tf.keras.layers.Conv2D(16,3,3,activation='tanh', padding="same"))
classifier.add(BatchNormalization())
classifier.add(tf.keras.layers.MaxPool2D(pool_size=(2,2), padding="same"))


classifier.add(tf.keras.layers.Conv2D(16,3,3,activation='tanh', padding="same"))
classifier.add(BatchNormalization())
classifier.add(tf.keras.layers.Conv2D(16,3,3,activation='tanh', padding="same"))

classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dense(units=64,activation='tanh'))
classifier.add(tf.keras.layers.Dropout(0.5))
classifier.add(tf.keras.layers.Dense(units=32,activation='tanh'))
classifier.add(tf.keras.layers.Dropout(0.5))
classifier.add(tf.keras.layers.Dense(units=16,activation='tanh'))
classifier.add(tf.keras.layers.Dropout(0.5))

classifier.add(tf.keras.layers.Dense(units=8,activation='softmax'))
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics='categorical_accuracy')

# %%
# Path'den dosya okunması
train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2,vertical_flip=True
                                                           ,zoom_range=0.2, horizontal_flip=True)
test_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, shear_range=0.2,vertical_flip=True
                                                              ,zoom_range=0.2, horizontal_flip=True)
train_set = train_data_gen.flow_from_directory(r'C:\Users\HP\Desktop\atelocerata\trainSet',
                                             target_size = (64,64), batch_size=10, class_mode='categorical')
test_set = test_data_gen.flow_from_directory(r'C:\Users\HP\Desktop\atelocerata\testSet', 
                                             target_size=(64,64), class_mode='categorical',batch_size=10)
classifier.fit_generator(train_set,epochs=700,validation_data=test_set)


# %%

test_set.reset()
pred=classifier.predict_generator(test_set, verbose=1)
pred[pred>=.5]=1
pred[pred<.5]=0
data_class=[]
test_labelsR=[]
test_labels=[]
for i in range(7):
    test_labels.extend(np.array(test_set[i][1]))
test_labels=np.asarray(test_labels)
#print(test_labels)
data_class=test_set.class_indices
CM=confusion_matrix(test_labels.argmax(axis=1),pred.argmax(axis=1))
print(CM)
print(CM.trace(),"/",test_labels.shape[0])
def get_key(val):
    for key, value in data_class.items():
         if val == value:
             return key
def prediction(x):
    data=classifier.predict(x.reshape(1,64,64,3)).argmax(axis=1)
    text=get_key(data)
    return text
#%%
with open('data_class.pkl','wb') as pickle_out:
    pickle.dump(data_class,pickle_out)
    
classifier.save("atelocerataModel")


#%%


























