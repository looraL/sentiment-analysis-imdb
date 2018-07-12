#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
an implementation of CNN model to predict pos/neg movie reviews

"""
import pandas as pd
import numpy as np
import re
import itertools
from collections import Counter
import os
#import nltk
#nltk.download('punkt')

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
#from Keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, Concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, concatenate, Lambda, BatchNormalization
from keras.layers import GlobalMaxPooling1D
from keras.models import Model, Sequential
from keras import backend as K
from keras.regularizers import l2
from keras.constraints import max_norm

import matplotlib.pyplot as plt

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from random import shuffle


prepare_data = False
construct_model = False
train = True
plot = True
check_prediction = False

MAX_SEQUENCE_LENGTH = 56
MAX_NB_WORDS = 10000 # number of words
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.2

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# 

def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads text data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]    
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # create dataframe with text and labels
    df_pos = pd.DataFrame.from_items([('text', positive_examples)])
    df_pos['label'] = 1    
    df_neg = pd.DataFrame.from_items([('text', negative_examples)])
    df_neg['label'] = 0
    # merge pos, neg reviews
    df = pd.concat([df_pos,df_neg], ignore_index=True)
    df['text'] = df.apply(lambda row: clean_str(row['text']), axis=1)
    #df['tokenized'] = df.apply(lambda row: nltk.word_tokenize(row[0]), axis=1)

    # randomly shuffle datasets into training and test sets
    #train, test = train_test_split(df, test_size=0.2)
    return df

if prepare_data:
    # data can be downloaded from http://www.cs.cornell.edu/people/pabo/movie-review-data
    # 5331 positive and 5331 negative processed sentences
    positive_data_file = "data/rt-polaritydata/rt-polarity.pos"
    negative_data_file = "data/rt-polaritydata/rt-polarity.neg"
    # text in lower_case already
    labels_index = {'pos': 1, 'neg': 0}
    df = load_data_and_labels(positive_data_file, negative_data_file)
    # pandas do the sampling randomly
    df_val = df.sample(frac=VALIDATION_SPLIT)
    df_train = df[~df.index.isin(df_val.index)]
    
    # count number of words in each sentence
    #df_len = df['text'].str.split().str.len()
    
    #    df_len.describe()
    #count    10661.000000
    #mean        20.387675
    #std          9.479101
    #min          1.000000
    #25%         13.000000
    #50%         20.000000
    #75%         27.000000
    #max         56.000000
    #Name: text, dtype: float64
    # set MAX_SEQUENCE_LENGTH = 50
    #df = df[df['text'].str.split().str.len() > 10]
    
    # follow this blog https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html    
    texts_train = df_train['text'].tolist()
    tokenizer_train = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer_train.fit_on_texts(texts_train)
    sequences_train = tokenizer_train.texts_to_sequences(texts_train)
    word_index_train = tokenizer_train.word_index
    print ('Found %s unique tokens for training set.' % len(word_index_train))
    x_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
    
    labels_train = df_train['label'].tolist()
    y_train = to_categorical(np.asarray(labels_train))
    
    
    texts_val = df_val['text'].tolist()
    tokenizer_val = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer_val.fit_on_texts(texts_val)
    sequences_val = tokenizer_val.texts_to_sequences(texts_val)
    
    word_index_val = tokenizer_val.word_index
    print ('Found %s unique tokens for validation set.' % len(word_index_val))
    x_val = pad_sequences(sequences_val, maxlen=MAX_SEQUENCE_LENGTH)
    
    labels_val = df_val['label'].tolist()
    y_val = to_categorical(np.asarray(labels_val))

    
    # compute an index mapping words to Glove embeddings
    #GLOVE_DIR = "../Glove"
    embeddings_index = {}
    f = open('Glove/glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    #print('total words in glove 6B: %s' % len(embeddings_index))
    
    # leverage our embedding index and word index to compute embedding matrix
    embedding_matrix = np.zeros((len(word_index_train) + 1, EMBEDDING_DIM))
    for word, i in word_index_train.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
  
if construct_model:      
    # load embedding matrix to an embedding layer
    # outputs a 3D tensor of shape (samples, sequence_length, embedding_dim)
    embedding_layer = Embedding(len(word_index_train) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    
    # simplified version of Yoon Kim's model
    # filter_size[3, 4, 5] gives val_acc = 74.8%
    # filter_size[7, 7, 7] gives val_acc = 75.7%
    # [7, 7, 7, 7] is chosen based on (Zhang et al. 2016) Section 4.3
    filter_sizes = [7, 7, 7, 7]
    convs = []
    for fsz in filter_sizes:
        l_conv = Conv1D(nb_filter=100,filter_length=fsz,activation='relu', kernel_regularizer=l2(0.05), kernel_constraint=max_norm(3.))(embedded_sequences)
        l_norm = BatchNormalization()(l_conv)
        #l_dropout = Dropout(0.5)(l_norm)
        # (Zhang et al.) proposed that globalMaxPooling gives the best performance, however it gives dimension error if I do the change
        # question on dimension[?,50,1,100], why 50 not 56
        #l_pool = GlobalMaxPooling1D()(l_norm)
        l_pool = MaxPooling1D(10)(l_norm)
        l_dropout = Dropout(0.5)(l_pool)
        l_flat = Flatten()(l_dropout)
        #convs.append(l_dropout)
        convs.append(l_flat)
    l_merge = concatenate(convs)
#    l_pool1 = MaxPooling1D(5)(l_merge)
#    l_dropout1 = Dropout(0.5)(l_cov1)
#    l_pool1 = MaxPooling1D(5)(l_cov1)
#    l_cov2 = Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.005))(l_pool1)
#    #l_dropout2 = Dropout(0.3)(l_cov2)
#    l_pool2 = MaxPooling1D(30)(l_cov2)   
#    l_flat = Flatten()(l_pool2)
#    l_dense = Dense(128, activation='relu', W_regularizer=l2(0.005))(l_flat)
#    pred = Lambda(lambda x: K.tf.nn.softmax(x))(l_dense)   
#    l_dense2 = Dense(2, W_regularizer=l2(0.005))(pred)
    l_dense = Dense(128, activation='relu', W_regularizer=l2(0.005))(l_merge)
    l_dropout2 = Dropout(0.5)(l_dense)
    #    pred = Lambda(lambda x: K.tf.nn.softmax(x))(l_dense)   
    #    l_dense2 = Dense(2, W_regularizer=l2(0.005))(pred)
    pred = Dense(1, activation='sigmoid', W_regularizer=l2(0.005))(l_dropout2)
    model = Model(sequence_input, pred)
    #l_dense2 = Dense(1, activation = 'softmax')(drop)
    # tf, keras version issue 
#    model = Sequential()    
#    model.add(model1)
#    model.add(Lambda(lambda x: K.tf.nn.softmax(x)))
#    model.add(Dense(2))
    
    # tutorial on optimizer: http://ruder.io/optimizing-gradient-descent/index.html#rmsprop
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
if train:
    # train model, batch_size is samples before gradient update, epoch is iterations over whole training set, split is
    #   percentage of training data to be used for testing rather than training
    #history = model.fit(X_train, y_train, validation_split=0.2, epochs=3, batch_size=50)
    
    y_train = y_train[:,1]
    y_val = y_val[:,1]
    
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=6, batch_size=32)
    score, acc = model.evaluate(x_val, y_val, batch_size=32)
    
if plot:
    # summarize history for accuracy
#    print(history.summary())
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #savefig('withoutDropout_acc')
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #savefig('withoutDropout_loss')
if check_prediction:
    # find false positives and false negatives
    # predict on test set then save incorrect predictions, note test set contains 25,000 records
    #incorrects = np.nonzero(model.predict(x_val).reshape((-1,)) != y_val)
    val_predict = model.predict(x_val)
    compare_val = pd.DataFrame({'y_val':y_val, 
                                'pred_val':val_predict.reshape((val_predict.shape[0],))})
    df_val['pred'] = val_predict.reshape((val_predict.shape[0],))

