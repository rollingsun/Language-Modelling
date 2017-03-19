
# coding: utf-8

# In[1]:

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from keras.optimizers import adam


# In[2]:

#Open the file and read contents
raw_text=open('Alice.txt').read()
raw_text=raw_text.lower()

#Mapping the characters to integers
chars=sorted(list(set(raw_text)))
char_to_int=dict((c,i) for i, c in enumerate(chars))

#Print the number of unique characters
print("Toal %d characters found. "%(len(chars)))
print(chars[1:10])

#Print total words in the vocabulary
print("Total %d words found in book" %len(raw_text))


# In[3]:

#Length of sequence to learn from is defined below
seq_length=40
X=[]
y=[]

for i in range(0, len(raw_text)-seq_length):
    seq_in=raw_text[i:i+seq_length]
    seq_out=raw_text[i+seq_length]
    X.append([char_to_int[char] for char in seq_in])
    y.append(char_to_int[seq_out])
    
#Here we have mapped 74 words sequence in X and the 75 character which is our target variable in Y

#total such patterns created
print("Total %d patterns are created"%len(X))


# In[ ]:

#First we must transform the list of input sequences into the form [samples, time steps, features] 
#expected by an LSTM network.

n_patterns=len(y)
X=np.reshape(X, (n_patterns, seq_length, 1))

# one hot encode the output variable
y = np_utils.to_categorical(y)


# In[ ]:

#Creating the LSTM Model

model=Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

# define the checkpoint
filepath="weights-improvement-second-run-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, nb_epoch=10, batch_size=200, callbacks=callbacks_list)



# In[ ]:



