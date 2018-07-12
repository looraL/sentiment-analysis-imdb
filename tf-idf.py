#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 19:07:28 2018

@author: lizhuoran
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from itertools import chain, repeat, islice


MAX_NB_WORDS = 20000
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

def pad_infinite(iterable, padding=0):
   return chain(iterable, repeat(padding))

def pad(iterable, size, padding=0):
   return islice(pad_infinite(iterable, padding), size)

test = [[2, 3, 5], [5, 6]]
test1 = pad(test, len(max(test,key=len)))


# data can be downloaded from http://www.cs.cornell.edu/people/pabo/movie-review-data
# 5331 positive and 5331 negative processed sentences
positive_data_file = "data/rt-polaritydata/rt-polarity.pos"
negative_data_file = "data/rt-polaritydata/rt-polarity.neg"
# text in lower_case already
labels_index = {'pos': 1, 'neg': 0}
df = load_data_and_labels(positive_data_file, negative_data_file)

# split into training and test sets
# pandas do the sampling randomly
df_val = df.sample(frac=VALIDATION_SPLIT)
df_train = df[~df.index.isin(df_val.index)]

texts_train = df_train.text
texts_test = df_val.text

# reference: Applied Text Analysis with Python by Tony Ojeda, Rebecca Bilbro, Benjamin Bengfort
# chapter 4

# One-hot embedding can be done with functions in Scikit-learn preprocessing module

# fit method expects an iterable or list of strings or file objects
# outputs a dictionary of the vocabulary:
# each individual document is transformed into an array with index tuple and values
# index tuple is the row (document ID) and the token ID from the dictionary, and value is the word count

# training set 80% and test set 20% of all sentences, training will get more features than test if not preset
# training has 16514 unique features, test has 7961 ones
count_vect   = CountVectorizer()
x_train_counts = count_vect.fit_transform(df_train.text)
x_test_counts = count_vect.transform(df_val.text)

## However, I was using Keras Tokenizer to prepare input data for the CNN,
## "sequences train" is somewhat result of one-hot encoding
## for consistency, would it be better to try out on this?  
## I came across an Error in the last step processing, haven't figured out a solution yet
#
#tokenizer_train = Tokenizer(nb_words=MAX_NB_WORDS)
#tokenizer_train.fit_on_texts(texts_train)
#
## this can also be viewed as "one-hot embedding", in the format of  list of list
## each inner list represents a document(sentence)
## for example, one document was converted into [2, 3525, 1312, 198, 108]
#
#sequences_train = tokenizer_train.texts_to_sequences(texts_train)
## padding and produce an array of size (#ofDocuments, total#ofWords)
#import itertools
#sequences_padded = np.array(list(itertools.izip_longest(*sequences_train, fillvalue=0))).T
#    
#enc = OneHotEncoder()
##AttributeError: 'OneHotEncoder' object has no attribute 'feature_indices_'?
## this worked on a smaller array, for example [[1, 2, 5], [2, 5, 9]]
#text_corpus = enc.transform(sequences_padded).toarray()

# tf-idf: “Term Frequency times Inverse Document Frequency”.
tfidf = TfidfTransformer()
# two steps: 1. fit() fit estimator to the data
#            2. transform() transform count-matrix to a tf-idf representation
x_train = tfidf.fit_transform(x_train_counts)
x_test = tfidf.transform(x_test_counts)

y_train = df_train.label
y_test = df_val.label

# default kernel='rbf'
#SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#  decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#  max_iter=-1, probability=True, random_state=None, shrinking=True,
#  tol=0.001, verbose=False)

clf = SVC(probability=True) 
#clf = SVC()
# problem: output prob. for two classes both equal to  0.5 ?? model problem 
clf.fit(x_train, y_train)

# predict and evaluate predictions
pred = clf.predict_proba(x_test)
pred = clf.predict(x_test)
np.mean(pred == y_test.label)
# no prediction of state "1"
matrix = confusion_matrix(y_test, pred)


#Building a pipeline:  make the vectorizer => transformer => classifier easier to work with
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, 
                                           random_state=42, max_iter=5, tol=None))])
text_clf.fit(df_train.text, df_train.label) 
predicted = text_clf.predict(df_val.text)
np.mean(predicted == df_val.label) 
# 0.7321763602251408

# grid search
parameters = {'vect__ngram_range': [(1, 1), (1, 5)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3)}
# n_job = -1 will detect # of cores installed and uses them all
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(df_train.text, df_train.label)
#sample_train = df_train.sample(frac=0.1)
#sample_test = df_val.sample(frac=0.1)
#gs_clf = gs_clf.fit(sample_train.text, sample_train.label)
gs_clf.best_score_ 
#0.724736
CV_result = gs_clf.cv_results_
#optimized para ngram_range = (1, 1); use_idf=True; clf_alpha=1e-3






