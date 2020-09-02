import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

from Bio import Seq, SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation
import networkx as nx
from itertools import islice
import re
import random

# import sklearn modules 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, make_scorer, roc_curve, auc
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# import the keras and word2vec modules
import tensorflow
from tensorflow.keras.constraints import max_norm
from keras.callbacks import EarlyStopping, Callback
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
from gensim.utils import *

from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Flatten, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras_self_attention import SeqSelfAttention


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        with open('model_history.txt', 'a') as f:
            stats = []
            stats.append(str(batch))
            stats.append('Optimizer,' + self.model.optimizer.__class__.__name__)
            stats.append('Batch_size,' + str(self.params['batch_size']))
            stats.append('accuracy,'+str(logs.get('accuracy')))
            stats.append('val_loss,'+str(logs.get('val_loss')))
            f.write(','.join(stats)+'\n')


# define functions to tokenize DNA and produce a 'sentence' of k-mer tokens 
def tokenize_dna(seq, ngram_size, ngram_stride):
    """
    Function to break up a DNA sequence into kmers
    
    Inputs:
    seq: str
    ngram_size: int
    ngram_stride: int
    """
    if ngram_size == 1:
        toks = list(seq) # for character level encoding of a dna sequence
    else:
        toks = [seq[i:i+ngram_size] for i in range(0, len(seq), ngram_stride)]
    return toks

# stitch the tokens together into a sentence 
def seq_sentence(seq, ngram_size, ngram_stride):
    """
    Function that stitches together DNA kmers into a sentence
    
    Inputs:
    seq: str
    ngram_size: int
    ngram_stride: int
    """
    toks = tokenize_dna(seq = seq, ngram_size = ngram_size, ngram_stride = ngram_stride)
    seq_sentence = ' '.join(toks)
    
    return seq_sentence

def seq_scrambler(seq):
    """ 
    Return the scrambled sequence of an input sequence. Ensures that the scrambled output returned has the same letter frequency
    but in a different order than the original sequence. Negative control for all models.
        
    Inputs:
    seq: str
    """
    scrambled_sequence = seq

    if len(set(seq)) == 1:
        return scrambled_sequence
    else:
        while scrambled_sequence == seq:
            chars = list(seq)
            random.shuffle(chars)
            scrambled_sequence = ''.join(chars)
    return scrambled_sequence

def train_sg_embedding(df_classify, ngram_size, ngram_stride, dim_embedding):
    """
    This function trains a skip-gram embedding over the corpus.
    
    Inputs:
    df_classify: df that contains a column called 'min_toehold_sequence' and 'quartile_rating'
                'min_toehold_sequence': str which is the minimal toehold sequence of length 59 nt
                'quartile_rating': integer encoded label to classify the top25% vs. the bottom 75%
    ngram_size: int
    ngram_stride: int
    dim_embedding: int
    """
    # convert the toehold sequence into a sentence of kmers and convert to lowercase (gensim word2vec prefers this to uppercase)
    df_classify['toehold_sentence'] = df_classify['min_toehold_sequence'].apply(lambda p: seq_sentence(seq=p, 
                                                                                                       ngram_size=ngram_size, 
                                                                                                       ngram_stride=ngram_stride).lower())

    # get a list of all toeholds
    toeholds = df_classify['min_toehold_sequence'].values.tolist()
    # scramble these sequences 
    scr_toes = [seq_scrambler(seq = p) for p in toeholds]
    # turn these scrambled seqs into sentences 
    scr_toes_sent = [seq_sentence(seq=p, ngram_size=ngram_size, ngram_stride=ngram_stride).lower() for p in scr_toes]
    df_classify['scrambled_sentence'] = scr_toes_sent
    
    # tokenize the k-mer sentence into 'words' and convert to a list of tokenized sentences
    df_classify['tokenized_toehold_sentence'] = df_classify['toehold_sentence'].apply(text_to_word_sequence) # break into separate words, separated by commas

    # get a list of all tokenized sentences with words separated by commas 
    tokenized_toehold_sentences = df_classify['tokenized_toehold_sentence'].values.tolist()
    max_words = 4**ngram_size + 5 # limit the vocabulary to this size (+4 for single letters and +1 for pad token)
    
    # train the skip-gram embedding model
    cores = multiprocessing.cpu_count() # Count the number of cores in a computer
    
    skip_seqmodel = Word2Vec(sentences = tokenized_toehold_sentences, 
                                 size = dim_embedding,
                                 window = 4, # size of context window about each word 
                                 sg = 1, # use skip-gram
                                 workers = cores - 1, # distribute training across multiple cores 
                                 max_final_vocab = max_words, 
                                 alpha = 0.1, # initial learning rate 
                                 min_alpha = 1e-4, # final learning rate 
                                 iter = 100, # number of passes over the document 
                                )
    
    # save this learned embedding for future use
    filename = 'toehold_sg_embeddings.txt'
    skip_seqmodel.wv.save_word2vec_format(filename, binary = False)

    # make a dictionary of embeddings 
    embeddings_index = {}
    f = open('toehold_sg_embeddings.txt', encoding = 'utf-8') # open saved embedding
    for line in f:
        values = line.split()
        # isolate the k-mer
        word = values[0] 
        # get its corresponding learned embeddings 
        coefs = np.asarray(values[1:])
        embeddings_index[word] = coefs # this is now a dictionary where each token has a ndarray of values 
    f.close()
    print('Embeddings trained and saved!')
    return df_classify, embeddings_index

def create_ds_splits(df_classify, embeddings_index, dim_embedding):
    """
    This function splits the df into training and testing datasets suitable for deep-learning using the pre-trained
    skip-gram embeddings. 
    
    Inputs:
    df_classify: df that contains a column called 'min_toehold_sequence' and 'quartile_rating'
                'min_toehold_sequence': str which is the minimal toehold sequence of length 59 nt
                'toehold_sentence': str which is the toehold sequence that is already split into kmers 
                'quartile_rating': integer encoded label to classify the top25% vs. the bottom 75%
                
    embeddings_index: dict - of embeddings given a particular tokenization scheme 
    """
    # list of toehold sentences
    toehold_sents = df_classify['toehold_sentence'].values.tolist()
    
    # keras only accepts integers in its embedding layer so we need to first create this integer encoding across the document
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(toehold_sents)
    tokenized_toeholds = tokenizer.texts_to_sequences(toehold_sents)
    
    word_index = tokenizer.word_index # contruct a dictionary of tokens with their respective ids
    max_length = max([len(s.split()) for s in toehold_sents]) # total number of tokens in the tokenized toehold sequence 
    
    # pad the sequences in-case they are not the same length 
    pad_toeholds = pad_sequences(tokenized_toeholds, maxlen = max_length)
    # training targets which in this case is classification 
    y = df_classify['quartile_rating'].values
    num_words = len(word_index) + 1 # add an extra word to account for unknowns 
        
    embedding_matrix = np.zeros((num_words, dim_embedding))
    # create an embedding weight matrix from the pre-generated and saved embedding
    for word, i in word_index.items():
        if i > num_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    x_train, x_test, y_train, y_test = train_test_split(pad_toeholds, y,
                                                        test_size = 0.10,
                                                        random_state = 0,
                                                        stratify = y,
                                                       )
    return x_train, x_test, y_train, y_test, embedding_matrix, max_length


def train_lstm(x_train, x_test, y_train, y_test, embedding_matrix, max_length):
    """
    Function that trains an LSTM model using the pre-trained skip-gram embedding
    
    Inputs:
    x_train: np array
    x_test: np array
    y_train: np array
    y_test: np array
    embedding_matrix: np array of shape num_words * dim_embedding
    max_length: int - maximum length of the sequence (in words)
    """
    kfold = StratifiedKFold(n_splits = 10, shuffle = False)
    es = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01)
    num_words = embedding_matrix.shape[0]
    dim_embedding = embedding_matrix.shape[1]
    
    accs = []
    aucs = []
    for train, valid in kfold.split(x_train, y_train):
        model = Sequential()
        embedding_layer = Embedding(input_dim = num_words, output_dim = dim_embedding, 
                            weights = [embedding_matrix],
                            input_length = max_length,
                            trainable = False)
        model.add(embedding_layer)
        model.add(Dropout(0.25))
        model.add(LSTM(units = 128, recurrent_dropout = 0.25, recurrent_regularizer = regularizers.l2(0.01)))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation = 'sigmoid')) # binary classification 
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])
        
        history = model.fit(x_train[train], y_train[train], 
                            validation_data=(x_train[valid], y_train[valid]),
                            callbacks = [es], epochs = 100, batch_size = 128, verbose = 0)
        
        score = model.evaluate(x_test, y_test, verbose = 0)
        accs.append(score[1])
        y_proba = model.predict_proba(x_test) # get the classification probs on the test set 
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label = 1)
        # Compute ROC area
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # plot training history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        
    return history, accs, aucs, model


def train_bid_lstm(x_train, x_test, y_train, y_test, embedding_matrix, max_length):
    """
    Function that trains a Bi-directional LSTM model using the pre-trained skip-gram embedding
    
    Inputs:
    x_train: np array
    x_test: np array
    y_train: np array
    y_test: np array
    embedding_matrix: np array of shape num_words * dim_embedding
    max_length: int - maximum length of the sequence (in words)
    """
    kfold = StratifiedKFold(n_splits = 10, shuffle = False)
    es = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01)
    num_words = embedding_matrix.shape[0]
    dim_embedding = embedding_matrix.shape[1]
    
    accs = []
    aucs = []
    for train, valid in kfold.split(x_train, y_train):
        model = Sequential()
        embedding_layer = Embedding(input_dim = num_words, output_dim = dim_embedding, 
                            weights = [embedding_matrix],
                            input_length = max_length,
                            trainable = False)
        model.add(embedding_layer)
        model.add(Dropout(0.25))
        model.add(Bidirectional(LSTM(units = 128, recurrent_dropout = 0.25, recurrent_regularizer = regularizers.l2(0.01) 
                  )))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation = 'sigmoid')) # binary classification 
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])
        
        history = model.fit(x_train[train], y_train[train], 
                            validation_data=(x_train[valid], y_train[valid]),
                            callbacks = [es], epochs = 100, batch_size = 128, verbose = 0)
        
        score = model.evaluate(x_test, y_test, verbose = 0)
        accs.append(score[1])
        y_proba = model.predict_proba(x_test) # get the classification probs on the test set 
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label = 1)
        # Compute ROC area
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # plot training history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        
    return history, accs, aucs, model

def train_self_atten_lstm(x_train, x_test, y_train, y_test, embedding_matrix, max_length):
    """
    Function that trains a Bi-directional LSTM model with Self-Attention using the pre-trained skip-gram embedding
    
    Inputs:
    x_train: np array
    x_test: np array
    y_train: np array
    y_test: np array
    embedding_matrix: np array of shape num_words * dim_embedding
    max_length: int - maximum length of the sequence (in words)
    """
    kfold = StratifiedKFold(n_splits = 10, shuffle = False)
    es = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01)
    num_words = embedding_matrix.shape[0]
    dim_embedding = embedding_matrix.shape[1]
    
    accs = []
    aucs = []
    for train, valid in kfold.split(x_train, y_train):
        model = Sequential()
        embedding_layer = Embedding(input_dim = num_words, output_dim = dim_embedding, 
                            weights = [embedding_matrix],
                            input_length = max_length,
                            trainable = False)
        model.add(embedding_layer)
        model.add(Bidirectional(LSTM(units = 128, recurrent_dropout = 0.25,
                                         return_sequences = True, recurrent_activation = 'relu',
                                         recurrent_regularizer = regularizers.l2(0.01) 
                  )))
        model.add(SeqSelfAttention(attention_width = 6, attention_activation = 'sigmoid'))
        model.add(Flatten()) # to get one output neuron 
        model.add(Dropout(0.25))
        model.add(Dense(1, activation = 'sigmoid')) # binary classification 
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])
        
        history = model.fit(x_train[train], y_train[train], 
                            validation_data=(x_train[valid], y_train[valid]),
                            callbacks = [es], epochs = 100, batch_size = 128, verbose = 0)
        
        score = model.evaluate(x_test, y_test, verbose = 0)
        accs.append(score[1])
        y_proba = model.predict_proba(x_test) # get the classification probs on the test set 
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label = 1)
        # Compute ROC area
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # plot training history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        
    return history, accs, aucs, model

def train_cnn_lstm(x_train, x_test, y_train, y_test, embedding_matrix, max_length):
    """
    Function that trains a CNN-LSTM model using the pre-trained skip-gram embedding
    
    Inputs:
    x_train: np array
    x_test: np array
    y_train: np array
    y_test: np array
    embedding_matrix: np array of shape num_words * dim_embedding
    max_length: int - maximum length of the sequence (in words)
    """
    kfold = StratifiedKFold(n_splits = 10, shuffle = False)
    es = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.01)
    num_words = embedding_matrix.shape[0]
    dim_embedding = embedding_matrix.shape[1]
    
    accs = []
    aucs = []
    for train, valid in kfold.split(x_train, y_train):
        model = Sequential()
        embedding_layer = Embedding(input_dim = num_words, output_dim = dim_embedding, 
                            weights = [embedding_matrix],
                            input_length = max_length,
                            trainable = False)
        model.add(embedding_layer)
        model.add(Dropout(0.25))
        model.add(Conv1D(filters = filters, kernel_size = 2, padding = 'same', activation = 'relu'))
        model.add(Conv1D(filters = filters, kernel_size = 3, padding = 'same', activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling1D(pool_size = 4))
        model.add(LSTM(units = 256, recurrent_dropout = 0.5, recurrent_regularizer = regularizers.l2(0.01)))
        model.add(Dense(1, activation = 'sigmoid')) # binary classification 
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])
        
        history = model.fit(x_train[train], y_train[train], 
                            validation_data=(x_train[valid], y_train[valid]),
                            callbacks = [es], epochs = 100, batch_size = 128, verbose = 0)
        
        score = model.evaluate(x_test, y_test, verbose = 0)
        accs.append(score[1])
        y_proba = model.predict_proba(x_test) # get the classification probs on the test set 
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label = 1)
        # Compute ROC area
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        # plot training history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        
    return history, accs, aucs, model



