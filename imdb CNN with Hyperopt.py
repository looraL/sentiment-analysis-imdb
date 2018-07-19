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

from bs4 import BeautifulSoup
import nltk
#nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords


from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
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

from hyperopt import hp, fmin, tpe,  STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials

import random
import string


prepare_data = False
load_exp_space = True
construct_model = True
train = False
plot = False
check_prediction = False

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000 # number of words in vocabulary
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

def clean_doc(string):
    string = BeautifulSoup(string).get_text()
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]"," ", string)
    string = re.sub(r"\\", "", string)      
    string = re.sub(r"\"", "", string)
    
    return string.encode('utf-8')


def clean_str(sentence, stop_words):
    """
    string cleaning for dataset
    All in lowercase
    """
    # Remove HTML syntax
    
    words = sentence.split()
	# remove punctuation from each token
    #table = words.maketrans('', '', string.punctuation)
    #words= [w.translate(table) for w in words]
    #words = re.sub(r"[^A-Za-z0-9(),!?\'\`]"," ", words)
    # remove remaining tokens that are not alphabetic
    words = [word for word in words if word.isalpha()]
    # set to lowercase
    words = [x.lower() for x in words]
    # filter out stop words
    words = [w for w in words if not w in stop_words]
    # filter out short tokens
    words = [word for word in words if len(word) > 1]
    
    return words



def prepare_train_val(data, train_size=1000):
    """
    Randomly shuffle data, tokenize and do word-level one-hot embedding on texts.
    Split the data into training set and validation set.
    Return x_train, y_train, texts_train, x_val, y_val, texts_val.
    """
    
    # clean string, format texts, labels into lists
    texts = []
    labels = []
    for ind in range(data.review.shape[0]):
        # Remove HTML syntax
        #text = BeautifulSoup(data.review[ind])
        text = data.review[ind]
        # encountering UnicodeDecodeError: unexpected end of data
        # https://stackoverflow.com/questions/24004278/unicodedecodeerror-utf8-codec-cant-decode-byte-0xc3-in-position-34-unexpect
        # if we remove all the conflicts
        #texts.append(clean_str(text.get_text().encode('ascii', 'ignore')))
        texts.append(clean_doc(text))
        labels.append(data.sentiment[ind])
        
    # pairwise shuffle on texts and labels 
    pack_texts_labels = list(zip(texts, labels))
    random.shuffle(pack_texts_labels)
    texts, labels = zip(*pack_texts_labels)
    texts = texts[:train_size]
    labels = labels[:train_size]
    
##################################################    
#    # word-level one-hot embedding with Keras
#    # configured to only take into account #top MAX_NB_WORDS of words
#    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
#    # this builds the word index
#    tokenizer.fit_on_texts(texts)
#    # this turns strings into lists of integer indices
#    # can also be viewed as "one-hot embedding", in the format of  list of list
#    # each inner list represents a document(sentence)
#    # for example, one document was converted into [2, 3525, 1312, 198, 108]
#    sequences = tokenizer.texts_to_sequences(texts)
#    
#    # hash words to their uniquely assigned integers.
#    word_index = tokenizer.word_index
#    # padding and produce an array of size (#ofDocuments, total#ofWords)
#    oneHot_result = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
###############################################    
    # word-level one-hot embedding without Keras
    # output np array(# of review, length of review), with word index as value
    # build an index of all tokens in the data.
     
    # Question: only include most common # of words? 
    # filter out stop_words
    # token_index equivalent to word_index in the commented keras one-hot embedding block
    stop_words = set(stopwords.words('english'))
    token_index = {}
    for review in texts:
        # strip punctuation and special characters from the review sentences.
        # then tokenize the reviews via the `split` method.
        review_cleaned = clean_str(review, stop_words)
        for word in review_cleaned:
            if word not in token_index:
                # Assign a unique index to each unique word
                token_index[word] = len(token_index) + 1
                # Note that 0 is not attributed to anything.
      
    # vectorization
    # only consider the first MAX_SEQUENCE_LENGTH words in each review.      
    # this is where we store the one-hot embedding results:
    oneHot_result = np.zeros((len(texts), MAX_SEQUENCE_LENGTH))
    
    for i, review in enumerate(texts):
        review = clean_str(review, stop_words)
        for j, word in list(enumerate(review))[:MAX_SEQUENCE_LENGTH]:
            index = token_index.get(word)
            oneHot_result[i, j] = index
###################################################    
            
     #format label
    labels = to_categorical(np.asarray(labels))
    
    #reverse_word_map = dict(map(sequences[0], word_index.items()))
    nb_validation_samples = int(VALIDATION_SPLIT * len(texts))
    
    x_train = oneHot_result[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    texts_train = texts[:-nb_validation_samples]
    x_val = oneHot_result[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]
    texts_val = texts[-nb_validation_samples:]

    return x_train, y_train, texts_train, x_val, y_val, texts_val, token_index

if prepare_data:
    # Kaggle IMDB dataset: https://www.kaggle.com/c/word2vec-nlp-tutorial/data
    # pd dataframe of size(25000, 3), columns: id, sentiment ,review(multiple sentences in one review)
    train_data = pd.read_csv( "data1/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
    #train_len = train.review.str.split().str.len()
    #train_len.describe()
    #count    25000.000000
    #mean       233.778560
    #std        173.721262
    #min         10.000000
    #25%        127.000000
    #50%        174.000000
    #75%        284.000000
    #max       2470.000000
    #Name: review, dtype: float64
    
    # set MAX_SEQUENCE_LENGTH = 1000
    
    # option: prepare_train_val(train, train_size = 500)
    x_train, y_train, texts_train, x_val, y_val, texts_val, token_index = prepare_train_val(train_data)

    # compute an index mapping words to Glove embeddings
    #GLOVE_DIR = "../Glove"
    embeddings_index = {}
    # (Zhang et al.) suggested 300d gives best performance
    f = open('Glove/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    # leverage our embedding index and word index to compute embedding matrix
    embedding_matrix = np.zeros((len(token_index) + 1, EMBEDDING_DIM))
    for word, i in token_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
            
    # originally in train section, this should only be changed once in preparation
    y_train = y_train[:,1]
    y_val = y_val[:,1]
    
    
if load_exp_space:
    #This is the parameter space to explore with hyperopt

    #Simply offers several discrete choices for numnber of hidden units and drop out rates for
    #a 2 or 3 layer MLP and also batch size
    
    #This can be expanded over other parameters and to sample from a distribution instead of a discrete choice
    #for # of units etc.
        
    space = {        
        'fsz': hp.choice('fsz', [3, 4, 5, 6, 7]),
        
        'conv_l2': hp.choice('conv_l2', [0.001, 0.005, 0.01, 0.1]),
        'dense1_l2': hp.choice('dense1_l2', [0.001, 0.005, 0.01, 0.1]),
        'pred_l2': hp.choice('pred_l2', [0.001, 0.005, 0.01, 0.1]),
                     
        'dropout1': hp.uniform('dropout1', 0, 1),        
        'dropout2': hp.uniform('dropout2', 0, 1),
        
        'dense_units': hp.choice('dense_units', [50, 100, 300]),
        
        'batch_size' : hp.choice('batch_size', [32,64,128]),
    
        'optimizer': hp.choice('optimizer', ['rmsprop','adam'])
        }


  
#if construct_model:      
def objective(params):
    # load embedding matrix to an embedding layer
    # outputs a 3D tensor of shape (samples, sequence_length, embedding_dim)
    embedding_layer = Embedding(len(token_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)
    
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    
    # trained on 800 reviews, validated on 200 reviews
    # dense layer immediate after concatenated convolutional layers
    # filter_size[7, 7, 7, 7] gives val_acc = 77.0%, dense 400 units, dropout = 0.5
    # filter_size[3, 4, 5] gives val_acc = 74.8%
    # filter_size[7, 7, 7] gives val_acc = 75.7%
    # [7, 7, 7, 7] is chosen based on (Zhang et al. 2016) Section 4.3
    # they suggested increasing the # of filters produces better result
    convs = []
    for i in range(3):    
        l_conv = Conv1D(nb_filter=100,filter_length=params['fsz'],activation='relu', kernel_regularizer=l2(params['conv_l2']))(embedded_sequences)
        l_norm = BatchNormalization()(l_conv)
        #l_dropout = Dropout(0.5)(l_norm)
        # (Zhang et al.) proposed that globalMaxPooling gives the best performance, however it gives dimension error if I do the change
        # question on dimension[?,50,1,100], why 50 not 56
        #l_pool = GlobalMaxPooling1D()(l_norm)
        l_dropout = Dropout(params['dropout1'])(l_norm)
        l_pool = MaxPooling1D(2)(l_dropout)
        
        # reduce the three-dimensional output to two dimensional for concatenation
        l_flat = Flatten()(l_pool)
        #convs.append(l_dropout)
        convs.append(l_flat)
    l_merge = concatenate(convs)
    #l_conv1 = Conv1D(100, 5, activation='relu', kernel_regularizer=l2(0.005))(l_merge)
    #l_dropout1 = Dropout(0.5)(l_conv1)
    #l_pool1 = MaxPooling1D(5)(l_dropout1)
#    l_cov2 = Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.005))(l_pool1)
#    #l_dropout2 = Dropout(0.3)(l_cov2)
#    l_pool2 = MaxPooling1D(30)(l_cov2)   
    #l_flat1 = Flatten()(l_pool1)
#    l_dense = Dense(128, activation='relu', W_regularizer=l2(0.005))(l_flat)
#    pred = Lambda(lambda x: K.tf.nn.softmax(x))(l_dense)   
#    l_dense2 = Dense(2, W_regularizer=l2(0.005))(pred)
    
    # why 128 filters? question not answered
    l_dense = Dense(params['dense_units'], activation='relu', W_regularizer=l2(params['dense1_l2']))(l_merge)
    # this dropout layer reduce "loss" from test set, observed from plots
    l_dropout2 = Dropout(params['dropout2'])(l_dense)
    #    pred = Lambda(lambda x: K.tf.nn.softmax(x))(l_dense)   
    #    l_dense2 = Dense(2, W_regularizer=l2(0.005))(pred)
    pred = Dense(1, activation='sigmoid', W_regularizer=l2(params['pred_l2']))(l_dropout2)
    model = Model(inputs= sequence_input, outputs=pred)
    #l_dense2 = Dense(1, activation = 'softmax')(drop)
    # tf, keras version issue 
#    model = Sequential()    
#    model.add(model1)
#    model.add(Lambda(lambda x: K.tf.nn.softmax(x)))
#    model.add(Dense(2))
    
    # tutorial on optimizer: http://ruder.io/optimizing-gradient-descent/index.html#rmsprop
    model.compile(loss='binary_crossentropy',
              optimizer=params['optimizer'],
              metrics=['accuracy'])
    
#if train:
    # train model, batch_size is samples before gradient update, epoch is iterations over whole training set, split is
    #   percentage of training data to be used for testing rather than training
    #history = model.fit(X_train, y_train, validation_split=0.2, epochs=3, batch_size=50)
    
    # epoch=2 generates best result
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=20, batch_size=params['batch_size'])
    score, acc = model.evaluate(x_val, y_val, batch_size=32)
    
    return {'loss': -acc, 'status': STATUS_OK, 'model': model }

if construct_model:
    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, trials=trials, max_evals=20)
    print (best)
    print (trials.best_trial)
    
    
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
    df_text_val = pd.Series(texts_val, name='text')
    df_label_val = pd.Series(y_val, name='label')
    df_val = pd.concat([df_text_val, df_label_val], axis=1)
    df_val['pred'] = val_predict.reshape((val_predict.shape[0],))
    false_pred = df_val.loc[((df_val['label'] < 0.5) & (df_val['pred'] > 0.5)) | 
                            ((df_val['label'] > 0.5) & (df_val['pred'] < 0.5))]
    
    writer = pd.ExcelWriter('compare_pred.xlsx')
    df_val.to_excel(writer,'all')
    false_pred.to_excel(writer, 'wrong')
    writer.save()
    
    


