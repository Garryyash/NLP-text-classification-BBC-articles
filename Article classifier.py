# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 09:25:00 2022

@author: caron
"""

#%% Imports

import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import seaborn as sns



#%% Constant
CSV_URL = 'https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv'




#%%
# EDA
# Step 1) Data Loading

df=pd.read_csv(CSV_URL)
df_copy = df.copy() #back-up


# Step2) Data Inspection
df.head(10)
df.tail(10)
df.info
df.describe()


df['category'].unique() #to get the unique target
df['text'][5]
df['category'][5]
df.duplicated().sum() # number of duplicated data
df[df.duplicated()]

sns.countplot(df.category)
# Step 3) Data Cleaning

df = df.drop_duplicates() #remove duplicated data
df.duplicated().sum() 
# NUmber of duplicated data is 0 now.

# remove html tags

text = df['text'].values # Features
category = df['category'].values # Target: category


for index,r in enumerate(text):
    text[index] = re.sub('.*?',' ',r)
    text[index] = re.sub('[^a-zA-Z]',' ',r).lower().split()

# Step 4) Features selection
# nothing to select


# Step 5) Data Preprocessing

#   1 Convert into lower case
            #Done
#   2 Tokenization

vocab_size = 15000
oov_token = 'OOV'


tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)

tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index
print(word_index)

train_sequences = tokenizer.texts_to_sequences(text) # to convert into numbers

#   3) Padding & truncating

length_of_text = [len(i) for i in train_sequences]
#np.median(length_of_text)

max_len = 333

padded_text = pad_sequences(train_sequences,maxlen=max_len,
                             padding='post',truncating='post')

#   4) One Hot Encoding for the target

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
category = ohe.fit_transform(np.expand_dims(category,axis=-1))

 #   5) Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(padded_text,category,test_size=0.3,
                                                 random_state=123)
 

X_train = np.expand_dims(X_train,axis=-1)
X_test = np.expand_dims(X_test,axis=-1)

#%% Model Development
# use LSTM layers,dropout, dense ,

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM , Dropout
from tensorflow.keras import Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Bidirectional,Embedding
from tensorflow.keras.layers import Masking

embedding_dim = 128

model = Sequential()
model.add(Input(shape=(333)))
model.add(Embedding(vocab_size,embedding_dim))
model.add(Bidirectional(LSTM(128,return_sequences=(True))))
model.add(Dropout(0.3))
model.add(Masking(mask_value=0)) #Masking Layer - Remove the 0 from padded data 
                                 # -> replace the 0 with the data values
model.add(Bidirectional(LSTM(128,return_sequences=(True))))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(np.shape(category)[1], activation='softmax'))           
model.summary()


plot_model(model)

# Wrapping the container
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

#CALLBACKS
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_FOLDER_PATH = os.path.join(os.getcwd(),'logs',log_dir)
tensorboard_callback = TensorBoard(log_dir=LOG_FOLDER_PATH)

# Model training
hist = model.fit(X_train,y_train,validation_data=(X_test,y_test),
                 batch_size=20,
                 epochs=50,
                 callbacks=[tensorboard_callback])

#%% hist keys
import matplotlib.pyplot as plt

hist.history.keys()
plt.figure()
plt.plot(hist.history['loss'],'r--',label='Training loss')
plt.plot(hist.history['val_loss'],label='Validation loss')
plt.legend()
plt.show()

plt.figure()
plt.plot(hist.history['acc'],'r--',label='Training acc')
plt.plot(hist.history['val_acc'],label='Validation acc')
plt.legend()
plt.show()

#%% Model Evaluation
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score

y_true = y_test

y_pred = model.predict(X_test)

y_true = np.argmax(y_true,axis=1)
y_pred = np.argmax(y_pred,axis=1) 

print(classification_report(y_true,y_pred))
# model score
print(accuracy_score(y_true,y_pred))

print(confusion_matrix(y_true,y_pred))

#%% Model saving
import os
MODEL_SAVE_PATH = os.path.join(os.getcwd(),'model.h5')
model.save(MODEL_SAVE_PATH)

import json
token_json = tokenizer.to_json()

TOKENIZER_PATH = os.path.join(os.getcwd(),'tokenizer_article.json')
with open (TOKENIZER_PATH,'w') as file:
    json.dump(token_json,file)

import pickle
OHE_PATH = os.path.join(os.getcwd(),'ohe.pkl')
with open (OHE_PATH,'wb') as file:
    pickle.dump(ohe,file)
#%% TensorBoard plot on browser

plot_model(model,show_shapes=True,show_layer_names=(True))


#%% Discussion
# Discuss your results
# Model achieved around 84% accuracy during training
# Recall and fi-score reports 87% and 84% respectively
# However the model starts to overfit after 2nd epoch
# EarlyStopping can be introduced in future to prevent overfitting
# Increase dropout rate to control overfitting
# Trying with different Model architecture for example BERT model, Transformer   
   
# 1) resuts ---> Discuss your results
# 2) give suggestion on how to improve the model
# 3) Gather evidence on what went wrong during training/model development











































