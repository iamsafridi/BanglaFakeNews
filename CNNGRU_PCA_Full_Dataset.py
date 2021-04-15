#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())
import numpy as np
import pandas as pd
from collections import defaultdict
import re,string,unicodedata


import sys
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, merge, Dropout
from keras.models import Model
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer,WordNetLemmatizer
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 100


# In[2]:


def clean_str(string):
    """
    Cleaning of dataset
    """
    string = re.sub(r"\\", "", string)    
    string = re.sub(r"\'", "", string)    
    string = re.sub(r"\"", "", string)    
    return string.strip().lower()


# In[3]:


true = pd.read_csv(r'H:\FakeNews\Authentic-48K.csv', na_values=['#NAME?'])
false = pd.read_csv(r'H:\FakeNews\Fake-1K.csv', na_values=['#NAME?'])


# In[4]:


data_train = pd.concat([false, true], ignore_index=True, sort=False)
data_train


# In[5]:


del data_train['date']
del data_train['articleID']
del data_train['domain']
del data_train['category']


# In[6]:


data_train


# In[7]:


nltk.download('stopwords')
stop = set(stopwords.words('bangla'))
punctuation = list(string.punctuation)
stop.update(punctuation)


# In[8]:


stemmer = PorterStemmer()
def stem_text(content):
    final_text = []
    for i in content.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text)  


# In[9]:


data_train.content = data_train.content.apply(stem_text)


# In[10]:


stemmer = PorterStemmer()
def stem_text(headline):
    final_text = []
    for i in headline.split():
        if i.strip().lower() not in stop:
            word = stemmer.stem(i.strip())
            final_text.append(word)
    return " ".join(final_text)  


# In[11]:


data_train.content = data_train.headline.apply(stem_text)


# In[12]:


# Input Data preprocessing
data_train = data_train
print(data_train.columns)
print('What the raw input data looks like:')
print(data_train[0:5])
texts = []
labels = []

for i in range(data_train.content.shape[0]):
    text1 = data_train.headline[i]
    text2 = data_train.content[i]
    text = str(text1) +""+ str(text2)
    texts.append(text)
    labels.append(data_train.label[i])
    
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))


# In[13]:


# Pad input sequences
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(labels),num_classes = 2)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)


# In[14]:


from sklearn.model_selection import train_test_split

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
x_train, x_test, y_train, y_test = train_test_split( data, labels, test_size=0.20, random_state=42)
x_test, x_val, y_test, y_val = train_test_split( x_test, y_test, test_size=0.50, random_state=42)
print('Size of train, validation, test:', len(y_train), len(y_val), len(y_test))

print('real & fake news in train,valt,test:')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))
print(y_test.sum(axis=0))


# In[15]:


from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization


# In[16]:


import numpy as np
from matplotlib import pyplot as plt
from keras.layers import Dense, Embedding, LSTM, GRU

get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


#Using Pre-trained word embeddings
GLOVE_DIR = "H:\FakeNews" 
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'bn_glove.39M.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    #print(values[1:])
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove.' % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH)


# In[18]:


embedding_vecor_length = 32
modelh = Sequential()
modelh.add(embedding_layer)
modelh.add(Dropout(0.2))
modelh.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
modelh.add(MaxPooling1D(pool_size=2))
modelh.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
modelh.add(MaxPooling1D(pool_size=2))
modelh.add(GRU(100, dropout=0.2, recurrent_dropout=0.2))
###
modelh.add(BatchNormalization())
###
#modelh.add(Dense(64, activation='relu'))
modelh.add(Dense(2, activation='softmax'))
modelh.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(modelh.summary())


# In[19]:


from tensorflow.keras.utils import plot_model
plot_model(modelh, to_file='modelh.png')


# In[20]:


history = modelh.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=128)
modelh.save('hybrid.h5')


# In[21]:


history_dict = history.history
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = history.epoch

#plt.figure(figsize=(9,6))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
#plt.title('Training and validation loss', size=10)
plt.xlabel('Epochs', size=20)
plt.ylabel('Loss', size=20)
plt.legend(prop={'size': 10})
plt.show()

#plt.figure(figsize=(9,6))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
#plt.title('Training and validation accuracy', size=10)
plt.xlabel('Epochs', size=20)
plt.ylabel('Accuracy', size=20)
plt.legend(prop={'size': 10})
#plt.ylim((0.5,1))
plt.show()


# In[22]:


# Test model 1
import seaborn as sns
test_preds = modelh.predict(x_test)
test_preds = np.round(test_preds)
correct_predictions = float(sum(test_preds == y_test)[0])
print("Correct predictions:", correct_predictions)
print("Total number of test examples:", len(y_test))
print("Accuracy of model1: ", correct_predictions/float(len(y_test)))

# Creating the Confusion Matrix
from sklearn.metrics import confusion_matrix, classification_report
x_pred = modelh.predict(x_test)
x_pred = np.round(x_pred)
x_pred = x_pred.argmax(1)
y_test_s = y_test.argmax(1)
cm = confusion_matrix(y_test_s, x_pred)
plt.matshow(cm, cmap=plt.cm.binary, interpolation='nearest')
plt.title('Confusion matrix - modelh')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')
plt.show()
sns.set(font_scale=1.4)#for label size
sns.heatmap(cm,annot=True,annot_kws={"size": 16},fmt='1f')# font size


# In[23]:


cv_report = classification_report(y_test,test_preds)
print(cv_report)


# In[24]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
pos_probs = x_pred
plt.plot([0, 1], [0, 1], linestyle='--')
fpr, tpr, _ = roc_curve(y_test_s, pos_probs)

# calculate AUC
auc = roc_auc_score(y_test_s, pos_probs)
print('AUC: %.3f' % auc)

plt.plot(fpr, tpr, linestyle='-', label=('AUC= %.2f' % auc))
plt.plot(fpr, tpr, marker='.', label='CNN+GRU')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()


# In[ ]:




