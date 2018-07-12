# Requires tf, numpy, keras, matplotlib libraries
# Drop "Merge" from keras.layers import in previous version since it is not used and not current keras usage
import numpy
import pandas as pd
#from keras.datasets import imdb
from tensorflow.python import keras
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, concatenate, Lambda, GlobalMaxPooling1D
from keras import backend as K
from keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold

import matplotlib.pyplot as plt

# preprocess data ourselves, without using Keras "load_data" method
from bs4 import BeautifulSoup
from os import listdir
import re
from nltk.corpus import stopwords

# use NLTK's punkt for sentence splitting
import nltk.data
nltk.download()

# solve ascii decode byte error
import sys  
reload(sys)  
sys.setdefaultencoding('utf8')

seed = 8
numpy.random.seed(seed)
MAX_NB_WORDS = 10000      # keep the top n words
MAX_SEQUENCE_LENGTH = 1000      # bound length of review

# preprocess data
# https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification
# load dataset, only keep the top n words, set rest to 0
# Note that imdb.load_data just loads one-hot coded integers, where "1" is the most common word, "2" the next most, etc
# See "Restore words from index","Preprocessing Data" to get words back https://medium.com/cityai/deep-learning-for-natural-language-processing-part-ii-8b2b99b3fa1e
# 0 does not stand for a specific word, it is used to denote "unknowns"
# optional: drop stopwords with "skip_top=n_words_to_skip"
# then try to Glove/word2vec that

#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=MAX_NB_WORDS)
#X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
############################################################################

# try with word2vec

# preprocess data
# Read data from files 
# train set has 25000 rows with three columns: id, sentiment(0 as neg/1 as pos), review
# set "delimiter=\t" to seperate fields by tabs
# set "quoting=3" to ignore double quotes

train = pd.read_csv( "data1/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )
#train.shape
#Out[24]: (25000, 3)
test = pd.read_csv( "data1/testData.tsv", header=0, delimiter="\t", quoting=3 )
#test.shape
#Out[25]: (25000, 2)
unlabeled_train = pd.read_csv( "data1/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3 )

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    
    # Remove HTML syntax
    review_text = BeautifulSoup(review).get_text()
    
    # Remove non-letters
    review_text = re.sub(r"[^A-Za-z0-9(),!?\'\`]"," ", review_text)
    
    # Convert words to lower case and split them
    words = review_text.lower().split()
    
    # Optional: Remove stop words-- high frequent words that do not carry much meaning, this may influence performance
    # haven't seen big difference in performance on this
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]    
    # Return a list of words
    return(words)

# word2vec expects a sentence as a list of words
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    # Function to split a review into single sentences. Returns a 
    # list of sentences, where each sentence is a list of words
    
    # Use NLTK tokenizer to split a review into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    
    # Loop through each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # skip empty sentence, o.w. call review_to_wordlist to get a list of words
        if len(raw_sentence) > 0:
            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    # Return a list of lists
    return sentences

# load data for word embedding
# this sentences list can be retrieved directly from saved result sentence.csv
# sentence is a list of list object, with each inner list representing a sentence
sentences = []  # Initialize an empty list of sentences
sentences_train = []
sentences_test = []
y_train = []
y_test = []

train2 = train.copy()
print "Parsing sentences from training set"
for review in train2["review"]:
    # encountering UnicodeDecodeError: unexpected end of data
    # https://stackoverflow.com/questions/24004278/unicodedecodeerror-utf8-codec-cant-decode-byte-0xc3-in-position-34-unexpect
    # if we remove all the conflicts, total number of sentences(labelled+unlabeled) decrease from around 857,000 to 795,000

    review = review.encode('ascii', errors='ignore')
    review_sentences = review_to_sentences(review, tokenizer)
    sentences_train += review_to_sentences(review, tokenizer)
    
   

print "Parsing sentences from unlabeled set"
for review in unlabeled_train["review"]:
    review = review.encode('ascii', errors='ignore')
    sentences_test += review_to_sentences(review, tokenizer)
sentences = sentences_train + sentences_test
    
#import csv
#with open ('sentences.csv', 'wb') as target:
#    wr = csv.writer(target, quoting=csv.QUOTE_ALL)
#    wr.writerow(sentences)
    
    
    
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#######################################################################################
# train the embedding model (this will take some time)
from gensim.models import word2vec
print "Training model..."

# better to use cython!! http://cython.readthedocs.io/en/latest/src/tutorial/cython_tutorial.html
# in this case, we can train the model using 4-8 threads
# or train the model on a computer with GPU
# will take almost 2 days otherwise...

# word embeddings with word2vec

# explanation on ALGOs can be found at:https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b
# or https://www.quora.com/What-are-the-continuous-bag-of-words-and-skip-gram-architectures
# skip gram: use current word to predict its neighbors(its context), e.g. give an input word, the network will tell the prob. for each word
#            in our vocabulary to be the "neighbor" of the word we chose
# CBOW: use contexts to predict the current word

# Set values for various parameters
num_features = 200    # Word vector dimensionality, e.g. the length of the dense
                      # vector to represent each token(word), hyperparameter to be tuned                      
min_word_count = 40   # Minimum word count, words with less than this count will be ignored when training                      
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                  
downsampling = 1e-3   # Downsample setting for frequent words

# the model is trained with skip-gram algorithm by defualt, can use CBOW by setting sg=0
model = word2vec.Word2Vec(sentences, workers=num_workers,
            size=num_features, min_count = min_word_count,
            window = context, sample = downsampling)
word_vectors = model.wv
#trained word vectors are independent from the methods used to train them, we can 
# represent them as a standalone structure KeyedVectors
# Word2Vec load functions are depreciated

# persist the word vectors to disk with...
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors

fname = get_tmpfile("200features_40minwords_10context.kv")
word_vectors.save(fname)
word_vectors = KeyedVectors.load(fname, mmap='r')

#model.most_similar("man")
#Out[26]: 
#[(u'woman', 0.6606647968292236),
# (u'lady', 0.6287051439285278),
# (u'lad', 0.5678527355194092),
# (u'guy', 0.5411106944084167),
# (u'men', 0.537365734577179),
# (u'person', 0.530378520488739),
# (u'monk', 0.5267703533172607),
# (u'businessman', 0.5263428688049316),
# (u'millionaire', 0.5201252102851868),
# (u'chap', 0.5184340476989746)]

################################################################
# Following the working example in "Word Embedded SVM.py"
# another option to load Stanford's pretrained Glove Embedding
from gensim.scripts.glove2word2vec import glove2word2vec
glove_input_file = 'Glove/glove.6B.300D.txt'
word2vec_output_file = 'glove.6B.300d.switch.txt'
glove2word2vec(glove_input_file, word2vec_output_file)

word2vecGloveswitch1 = 'glove.6B.300d.switch.txt'
model = KeyedVectors.load_word2vec_format(word2vecGloveswitch1, binary=False)

# Convert movie reviews of words into lists of 300-dimension word embedding vectors
vocab = model.vocab.keys()
vectors_train=[]
vectors_test=[]
for r in range(len(sentences_test)):
    new_temp=[0]*300
    count=0
    for w in range(len(sentences_test[r])):
        if train[r][w] in vocab:
            count=count+1
            new_temp=new_temp+model[sentences_test[r][w]]
    vectors_train.append(new_temp/count)


# construct model
# apply keras default embeddings: please explain what this does precisely
# Note that restricting to the top N words isn't necessary with a word2vec type embedding, since the embedding
#   itself will drop all words for which a pre-trained "meaning" doesn't exist
# Try using a word2vec or similar embedding; I believe this just one-hot codes

# explanations: this keras embedding layer maps the integer inputs to the vectors found at the corresponding index, for example [1, 2] will map to [embedding[1], embedding[2]]
# For its iput, each word is represented by a unique integer, ie.one-hot encoding, this step can be done with Keras tokenizer
# the embedding layer starts with random weights and will learn an embedding for all words in the vacabulary
# this layer serves as the first hidden layer of the network, thus we need to specify a few arguments for the input

# input arguments
# input_dim: MAX_NB_WORDS   -- the size of the words in a text review, with paddings
# output_dim: 100 in this case -- the size of the output vectors for each word 
# input_length: MAX_SEQUENCE_LENGTH  --length of input sequence, eg. average/max length of all input documents, with paddings
# tainable: set True so that weights will be kept updated during the training process
# can also load pretrained weights such as trained embedding matrix from Glo, in this case trainable can be set False
# structure of embedding layer
embedding_layer = Embedding(MAX_NB_WORDS,
                            100,
                            #weights=[word_vectors],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=True)

# Make sure you know the data type and structure of sequence input here; imdb data makes it easy but often this
#   step requires some thought
###################################################################################
# input: A shape tuple eg.(50,), not including the batch size
# using None allows the network to handle any sized batch
# the following line indicates input will be batches of MAX_SEQUENCE_LENGTH-dimensional vectors
# MAX_SEQUENCE_LENGTH is the length of each review in keras IMDB dataset. Some used length of each review sentence(with paddings)
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
#The result of the embedding layer is a 3-dimensional tensor of shape [None, max_sequence_length, output_dim(embedding size)]
embedded_sequences = embedding_layer(sequence_input)

# simplified version of Yoon Kim's model
# PROVIDE NOTES ON WHY FILTER SIZE, # OF DENSE LAYERS, AND PARTICULAR REGULARIZER ARE BEING USED; cites to lit are fine
###################################################################################
# as explained in Chapter4.3 in Yoon Kim's CNN for Sentence Classification in contrast with Kalchbrenner et al.(2014), 
# Apply different filter size to learn complementary features from a same "region", 
# details in Zhang et al. "practioner's guide" Section 4.3, suggested setting filters with size[7, 7, 7, 7] achieves best results
# and they also suggested better performance will be achieved with more filters, they used 400 filters in total
filter_sizes = [3, 4, 5]
convs = []
# provide notes on each line here; it is a bit non-standard as written
for fsz in filter_sizes:
    # "kernel" refers to the sliding window, also denoted as "filter" as above
    # for this 1-D convolutional layer, we use "filters" number of  of filters, each with size "kernel_size * kernel_size"
    # and here we use 'relu' max(0, a) where a = Wx + b as activation, filters have pixel values 0 or 1
    # for each of these filters, slide it through the whole vector matrix and do element-wise multiplication
    #
    # ReLU(Krizhevsky et al., 2012), also referenced http://cs231n.github.io/neural-networks-1/ in the ReLU section: 
    #       pros: 1. accelerate convergence of stochastic gradient descent
    #             2.performance wise better than tahn and sigmoid,      
    #       cons: dead neuron(can be solved by initializing bias/ use maxout), https://medium.com/joelthchao/how-dead-neurons-hurt-training-5fc127d8db6a
    # However, "Iden" (Identity function without activation function ) gave best result(Zhang et al., 2016 Section 4.5)
    #
    # kernel_regularizer applies to kernel weights matrix
    # use paddings if needed, zero-padded in this case
    
    # Yoon Kim used 100 filters, so did Zhang, I didn't find detailed reason for this. 100 is considered large enough? 
    l_conv = Conv1D(filters=128, kernel_size=fsz,activation='relu', kernel_regularizer=l2(0.003))(word_vectors)
    # apply dropout to reduce overfitting
    # prevent co-adaptation of hidden units by setting 0 with prob. p in forward propagation
    # (Hinton et al., 2012)
    l_dropout = Dropout(0.5)(l_conv)
    #  try 1-max pooling
    # to capture the most important feature -- one with the highest value, for each feature map
    # to reduce number of parameters. Try dropout after pooling? Dropout will be more powerful in this case
    l_pool = MaxPooling1D(5)(l_dropout)  
    # flatten as fully connected(FC) layer, in this way feature maps will be vectorized as (x1, x2, ...) each x is a feature map
    # with the fully connected layer, we have all these features combined
    l_flat = Flatten()(l_pool)
    #convs.append(l_dropout)
    convs.append(l_flat)
    # repeat the process to extract features(computing in parallel, not increasing depth) with other filter widths, a nice plot on this procedure can be found in Yoon Kim's paper
l_merge = concatenate(convs)
# to reduce overfitting, I commented out the following few lines
# they were used to deepen the network with two additional convolutional layers
#    l_cov1= Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.005))(l_merge)
#    #l_dropout1 = Dropout(0.3)(l_cov1)
#    l_pool1 = MaxPooling1D(5)(l_cov1)
#    l_cov2 = Conv1D(128, 5, activation='relu', kernel_regularizer=l2(0.005))(l_pool1)
#    #l_dropout2 = Dropout(0.3)(l_cov2)
#    l_pool2 = MaxPooling1D(30)(l_cov2)   
#l_flat = Flatten()(l_merge)
## WHY ARE YOU USING A 128-DIMENSION SPACE IN EACH DENSE LAYER?  NOT NECESSARILY WRONG BUT WANT YOU TO KNOW WHY
# ?? 128 units, not knowing why
# WHY ARE YOU USING L2 TO REGULARIZER RATHER THAN DROPOUT OR NOTHING AT ALL?
# l2 regularizer gave better result in loss reduce
# WHY NOT JUST USE FEWER EPOCHS TO AVOID OVERFITTING RATHER THAN THIS DROPOUT?
# there was a tendency of increasing learning rate after 2 epoches, so I set it to 3
# There are answers to all of these questions in the literature - but I want to make sure you understand
#   why we are doing certain things at each stage

# Why 128 filters? 
l_dense = Dense(128, activation='relu', W_regularizer=l2(0.005))(l_merge)
#    pred = Lambda(lambda x: K.tf.nn.softmax(x))(l_dense)   
#    l_dense2 = Dense(2, W_regularizer=l2(0.005))(pred)
pred = Dense(1, activation='sigmoid', W_regularizer=l2(0.005))(l_dense)
model = Model(sequence_input, pred)
#l_dense2 = Dense(1, activation = 'softmax')(drop)
# tf, keras version issue 
#    model = Sequential()    
#    model.add(model1)
#    model.add(Lambda(lambda x: K.tf.nn.softmax(x)))
#    model.add(Dense(2))

# tutorial on optimizer: http://ruder.io/optimizing-gradient-descent/index.html#rmsprop
# binary cross-entropy with sigmoid layers is generally fine with multiclass classification but look into this
# adam as optimizer is a particular gradient descent optimizer, metric is what to show as measure of success
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])


# train model, batch_size is samples before gradient update, epoch is iterations over whole training set, split is
# percentage of training data to be used for testing rather than training
history = model.fit(X_train, y_train, validation_split=0.2, epochs=3, batch_size=50)

# plot
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
