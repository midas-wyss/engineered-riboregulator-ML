# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import sys
sns.set_style('ticks', {'axes.grid': False, 
                        'grid.linestyle': '', 
                        'font.family':'sans-serif', 
                        'font.sans-serif':'Myriad Pro',
                        'text.color': '0',
                        'xtick.color': '0',
                        'ytick.color': '0'
                           })
import umap

sys.path.insert(1, '/work/06658/pramesh/maverick2/notebooks/')
import NuSpeak_base as nuspeak

from Bio import Seq, SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation
import networkx as nx
from itertools import islice
import re
import random
from pathlib import Path

import torch
from torch.utils.data.sampler import WeightedRandomSampler
from fastai import *
from fastai.text import *
from fastai.text.interpret import *
from fastai.callbacks import *
__all__ = ['OverSamplingCallback']

# import look-ahead optimizer in lieu of Adam
from ranger import Ranger
optar = partial(Ranger)

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# import sklearn modules 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import matthews_corrcoef as mcc

label_encoder = LabelEncoder()
kfold = StratifiedKFold(n_splits = 5, shuffle = True)

# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


path = Path('workdir/excels')
path2 = Path('workdir/encoders')
pltpath = 'workdir/results/'


# define parameters
n_trials = 5
ngram_size = 3
ngram_stride = 3
embedding_size = 400
test_frac = 0.10

def pred_class(seq, learn_cf):
    cat_fwd, y_fwd, arr_fwd = learn_cf.predict(seq)
    p1f = to_np(arr_fwd)[1]
    yf = to_np(y_fwd)
    cls_fwd = cat_fwd.obj
    return cls_fwd

# define a tokenizer class that calls up Fastai's base tokenizer
class GenomicConstantTokenizer2(BaseTokenizer):

    def __init__(self, lang = 'en', kmer = ngram_size, stride = ngram_stride):
        self.lang = lang
        self.kmer = kmer
        self.stride = stride

    def tokenizer(self, t):
        t = t.upper() # ensure all uppercase 
        if self.kmer == 1:
            toks = list(t) # trivial case when the kmer is of length 1 
        else:
            toks = [t[i:i+self.kmer] for i in range(0, len(t), self.stride) if len(t[i:i+self.kmer]) == self.kmer]   
        return toks # list of tokens 
    
    def add_special_cases(self, toks):
        pass

# generate a series of random toeholds to serve as the base for the synthetic language model

# convert the switches into toehold sensors given the switch sequence 
# and the start_codon and the rbs site 
def turn_switch_to_toehold(switch, rbs = 'AACAGAGGAGA', start_codon = 'ATG'):
    stem1 = Seq(switch[24:30]).reverse_complement().__str__()
    stem2 = Seq(switch[12:21]).reverse_complement().__str__()
    toehold =  switch + rbs + stem1 + start_codon + stem2
    return toehold

# check for any stop codons that occur after the start and remove these from the pool 
## revised so that we only consider no in-frame stops 
def check_for_stop(toehold): 
    stop_codons = ['TAG', 'TAA', 'TGA']
    bad_locations = [47, 50, 53, 56]
    search = False
    for stop in stop_codons:
        for bad_loc in bad_locations:
            stop_index = toehold.find(stop, bad_loc, bad_loc + 3) # val, start search, end search
            search_test = stop_index == bad_loc
            search = search | search_test
    return search


rand_switches = []
for i in range(4*10**6):
    temp = nuspeak.rand_seq(30)
    rand_switches.append(temp)

df_randLM = pd.DataFrame()
df_randLM['switch'] = rand_switches
df_randLM['min_toehold_sequence'] = df_randLM['switch'].apply(lambda x: turn_switch_to_toehold(x, rbs = 'AACAGAGGAGA', start_codon = 'ATG'))
# we now have a dataframe of all possible toeholds
start_codon = 'ATG'
toeholds = df_randLM['min_toehold_sequence'].values.tolist()
# remove all toeholds where there are start codons before the intended one starting at position 47
no_start = [x for x in toeholds if x.index(start_codon) == 47]
no_stop = [x for x in no_start if not check_for_stop(x)]

# create a new dataframe of randomly generated toeholds for the LM training.
df_randLM = pd.DataFrame()
df_randLM['min_toehold_sequence'] = no_stop


# load in the toehold dataset and preprocess for ML to create a df for classification
# based on the ON/OFF ratios of the sensors

df = pd.read_csv(path/'newQC_toehold_data.csv', comment = '#')

# rename the columns 
rename_dict = {
    "Unnamed: 0" : "toehold_id",
    "onoff_value" : "delta_onoff",
    "onoff_qc" : "delta_qc_onoff",
    "switch_sequence" : "min_toehold_sequence"
}

# clean up df to get rid of NaN and low qc reads 
df = df.rename(columns = rename_dict)
df = df.dropna() # throw out nan's 
ngs_qc_onind = df['on_qc'] >= 1.1 # keep all the acceptable reads for the ON
ngs_qc_offind = df['off_qc'] >= 1.1 # keep all the acceptable reads for the OFF
df = df.loc[ngs_qc_onind & ngs_qc_offind, :]

# bin the toeholds by their ON/OFF ratio into quartlies for the classifier 
df['ON/OFF quartile'] = pd.qcut(df['delta_onoff'], 
                                      q = 4, 
                                      labels = ['Bottom', '25-50', '50-75', 'Top',
                                               ])

ind_top = df['ON/OFF quartile'] == 'Top' # find the top 25% of toeholds
ind_bottom = df['ON/OFF quartile'] != 'Top' # find the bottom 75% of toeholds
df_best_toeholds = df.loc[ind_top, :] # slice out the top 25%
df_bad_toeholds = df.loc[ind_bottom, :] # slice out the bottom 75%

df_best_toeholds['Toehold Rating'] = 'Good'
df_bad_toeholds['Toehold Rating'] = 'Bad'
df_classify = pd.concat([df_best_toeholds, df_bad_toeholds], axis = 0)
df_classify['scrambled_toehold'] = df_classify['min_toehold_sequence'].apply(lambda x: nuspeak.seq_scrambler(x))
df_classify['shufftok_toehold'] = df_classify['min_toehold_sequence'].apply(lambda x: nuspeak.sent_scrambler(seq = x, 
                                                                                       ngram_size = ngram_size, 
                                                                                       ngram_stride = ngram_stride))
df_classify = df_classify.sample(frac = 1).reindex() 
# find all random sequences 
substring1 = 'random'
df_classify['indices'] = df_classify['sequence_id'].str.find(substring1)
ind = df_classify['indices'] >= 0
ind2 = df_classify['indices'] < 0

df_random = df_classify.loc[ind,:]
df_random['class'] = 'random'
df_random = df_random.drop('indices', axis = 1)

df_classify = df_classify.loc[ind2,:]
df_classify = df_classify.drop('indices', axis = 1)

# find all human_tf sequences 
substring2 = 'human'
df_classify['indices'] = df_classify['sequence_id'].str.find(substring2)
ind = df_classify['indices'] >= 0
ind2 = df_classify['indices'] < 0

df_human = df_classify.loc[ind,:]
df_human['class'] = 'human'
df_human = df_human.drop('indices', axis = 1)

df_classify = df_classify.loc[ind2,:]
df_classify = df_classify.drop('indices', axis = 1)

# the rest are all viral sequences 
df_viral = df_classify
df_viral['class'] = 'viral'

df_classify = pd.concat([df_random, df_human, df_viral], axis = 0)


# lets create the training and validation sets for the random toehold lM
valid_df, train_df = nuspeak.split_gendata(df_randLM, test_frac)
train_df['is_train'] = 1
valid_df['is_train'] = 0

tokenizer = Tokenizer(tok_func = GenomicConstantTokenizer2, 
                      pre_rules = [], post_rules = [],
                      special_cases = ['xxpad'])


# forward language 
toehold_LMf = nuspeak.GenomicTextLMDataBunch.from_df(path = path, train_df = train_df, valid_df = valid_df,
                                                   tokenizer = tokenizer, text_cols = 'min_toehold_sequence', 
                                                  label_cols = 1, bs = 128, backwards = False, bptt = 25) # create a forward language model 
# reverse language
toehold_LMr = nuspeak.GenomicTextLMDataBunch.from_df(path = path, train_df = train_df, valid_df = valid_df,
                                                   tokenizer = tokenizer, text_cols = 'min_toehold_sequence', 
                                                  label_cols = 1, bs = 128, backwards = True, bptt = 25) # create a backwards language model
config_fwd = dict(emb_sz = embedding_size, 
              n_hid = 1552, 
              n_layers = 4, 
              pad_token = 0, 
              qrnn = True,
              bidir = False, 
              output_p = 0.2, 
              hidden_p = 0.20, 
              input_p = 0.30, 
              embed_p = 0.1, 
              weight_p = 0.25, 
              tie_weights = True, 
              out_bias = True)

config_rev = dict(emb_sz = embedding_size, 
              n_hid = 1552, 
              n_layers = 4, 
              pad_token = 0, 
              qrnn = True,
              bidir = False, 
              output_p = 0.2, 
              hidden_p = 0.20, 
              input_p = 0.30, 
              embed_p = 0.1, 
              weight_p = 0.25, 
              tie_weights = True, 
              out_bias = True)


classify_config = dict(
    emb_sz = embedding_size,
    n_hid = 1552, # multiply by two to account for bidirectionality if True
    n_layers = 4,
    pad_token = 0,
    qrnn = True,
    bidir = False,
    output_p = 0.5, # standard dropout applied to activations in the linear head
)

drop_mult = 1.0 # multiplier across all dropouts 
wd = 0.1 # strong l2 regularization

# initialize the language model learners 
learn_fwd = nuspeak.NuSpeak_learner(data = toehold_LMf, 
                                      arch = AWD_LSTM, 
                                      config = config_fwd, 
                                      drop_mult = drop_mult, 
                                      wd = wd)

learn_rev = nuspeak.NuSpeak_learner(data = toehold_LMr, 
                                      arch = AWD_LSTM, 
                                      config = config_rev, 
                                      drop_mult = drop_mult, 
                                      wd = wd)

# save the vocabs from the language models 
np.save(path2/'toehold_LMf_vocab.npy', toehold_LMf.vocab.itos)
np.save(path2/'toehold_LMr_vocab.npy', toehold_LMr.vocab.itos)


# save the vocabs from the language models 
np.save(path2/'toehold_LMf_vocab.npy', toehold_LMf.vocab.itos)
np.save(path2/'toehold_LMr_vocab.npy', toehold_LMr.vocab.itos)

# train the forward model 
learn_fwd.unfreeze()
schedf = nuspeak.FlatCosAnnealScheduler(learn_fwd, lr = 5e-3, tot_epochs = 15, moms = (0.8, 0.7));
learn_fwd.callbacks.append(schedf)
learn_fwd.fit(15)
learn_fwd.save('toe_LMf')
learn_fwd.save_encoder('toe_LMf_enc')

# train the reverse model 
learn_rev.unfreeze()
schedr = nuspeak.FlatCosAnnealScheduler(learn_rev, lr = 5e-3, tot_epochs = 15, moms = (0.8, 0.7));
learn_rev.callbacks.append(schedr)
learn_rev.fit(15)
learn_rev.save('toe_LMr')
learn_rev.save_encoder('toe_LMr_enc')

#
sample_fracs = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

y_targets = df_classify['Toehold Rating']
X_samples = df_classify.drop('Toehold Rating', axis = 1)

scores = []
aucs = []
fracs = []
mccs = []
mccs_c1 = []
mccs_c2 = []
c1_scores = [] # for the control shuffled tokens
c2_scores = [] # for the control shuffled characters
tprs = []
aucs = []
trials = []

# here we are also sampling the amount of data needed for the LM pretraining 
for sampling in sample_fracs:
    
    for trial in range(n_trials):

        # first prepare the dataset for testing
        X_withheld, X_test, y_withheld, y_test = train_test_split(X_samples, 
                                                            y_targets, 
                                                            train_size = 1 - test_frac, 
                                                            shuffle = True, 
                                                            stratify = y_targets)
        
        # now partition the remainder for testing and validation 
        X_train, X_valid, y_train, y_valid = train_test_split(X_withheld, 
                                                    y_withheld, 
                                                    train_size = 1 - test_frac, 
                                                    shuffle = True, 
                                                    stratify = y_withheld)
            
        df_train = pd.concat([X_train, y_train], axis = 1)
        df_train['set'] = 'train'
        
        df_valid = pd.concat([X_valid, y_valid], axis = 1)
        df_valid['set'] = 'valid'
        
        df_test = pd.concat([X_test, y_test], axis = 1)
        df_test['set'] = 'test'
        
        toehold_df = pd.concat([df_train, df_valid], axis = 0)
        toehold_df = toehold_df.sample(frac = 1).reindex() # shuffle up
        toehold_df = toehold_df.sample(frac = sampling) # sub-sample the df
        
        train_df = toehold_df[toehold_df.set == 'train']
        valid_df = toehold_df[toehold_df.set == 'valid']
        test_df = df_test # evaluate on the original test set 

        
        # create a databunch for feeding into the NLP routine using the same tokenizer as the language model 
        toehold_dbf = nuspeak.GenomicTextClasDataBunch.from_df(path = path, train_df = df_train, valid_df = df_valid,
                                                test_df = df_test, tokenizer = tokenizer, vocab = fwd_model_vocab,
                                                text_cols = 'min_toehold_sequence', label_cols = 'Toehold Rating',
                                                bs = 128, backwards = False)

        toehold_dbr = nuspeak.GenomicTextClasDataBunch.from_df(path = path, train_df = df_train, valid_df = df_valid,
                                                test_df = df_test, tokenizer = tokenizer, vocab = rev_model_vocab,
                                                text_cols = 'min_toehold_sequence', label_cols = 'Toehold Rating',
                                                bs = 128, backwards = True)

        learn_cf = nuspeak.NuSpeak_classifier_learner(data = toehold_dbf, 
                                               arch = AWD_LSTM, 
                                               config = classify_config, 
                                               drop_mult = drop_mult,
                                               clip = None, wd = 0.1, bptt = 25)
        
        learn_cf.load_encoder('/work/06658/pramesh/maverick2/excels/models/toe_LMf_enc')
        learn_cf.fit_one_cycle(9, max_lr = 5e-2)
        
        learn_cf.freeze_to(-2)
        learn_cf.fit_one_cycle(5, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.95))
        
        learn_cf.freeze_to(-3)
        learn_cf.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.95))
        
        learn_cf.unfreeze()
        learn_cf.fit_one_cycle(8, max_lr = 2.5e-3)

        
        data_classify_testf = nuspeak.GenomicTextClasDataBunch.from_df(path = path, train_df = df_train, valid_df = df_test,
                                                tokenizer = tokenizer, vocab = fwd_model_vocab,
                                                text_cols = 'min_toehold_sequence', label_cols = 'Toehold Rating',
                                                bs = 128, backwards = False)
        # assign this data to the trained learner 
        learn_cf.data = data_classify_testf
        # compute metrics 
        preds, _, _ = learn_cf.get_preds(ordered = True, with_loss = True)
        roc_aucf = roc_auc_score(df_test['Toehold Rating'], preds[:,1])
        acuf, mccf = nuspeak.get_metrics(learn_cf, return_metrics = True)
        
        
        learn_cr = dulm.dna_classifier_learner(data = toehold_dbr, 
                                               arch = AWD_LSTM, 
                                               config = classify_config, 
                                               drop_mult = drop_mult,
                                               clip = None, wd = 0.1, bptt = 25)
        learn_cr.load_encoder('/work/06658/pramesh/maverick2/excels/models/toe_LMr_enc')
        learn_cr.fit_one_cycle(9, max_lr = 5e-2)
        
        learn_cr.freeze_to(-2)
        learn_cr.fit_one_cycle(5, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.95))
        
        learn_cr.freeze_to(-3)
        learn_cr.fit_one_cycle(5, slice(1e-3/(2.6**4),1e-3), moms=(0.8,0.95))
        
        learn_cr.unfreeze()
        learn_cr.fit_one_cycle(8, max_lr = 2.5e-3)
        
        # define the test set data bunch
        data_classify_testr = dulm.GenomicTextClasDataBunch.from_df(path = path, train_df = df_train, valid_df = df_test,
                                                tokenizer = tokenizer, vocab = rev_model_vocab,
                                                text_cols = 'min_toehold_sequence', label_cols = 'Toehold Rating',
                                                bs = 128, backwards = True)
        # assign this data to the trained learner 
        learn_cr.data = data_classify_testr
        # compute metrics 
        preds, _, _ = learn_cr.get_preds(ordered = True, with_loss = True)
        roc_aucr = roc_auc_score(df_test['Toehold Rating'], preds[:,1])
        acur, mccr = nuspeak.get_metrics(learn_cr, return_metrics = True)
        
        test_df['shufftok_class'] = test_df['shufftok_toehold'].apply(lambda x: pred_class(x, learn_cf))
        y_test_true = test_df['Toehold Rating']
        y_test_shufftok = test_df['shufftok_class']
        mcc_c1 = mcc(y_test_true, y_test_shufftok)
        c1_scores.append(accuracy_score(y_test_true, y_test_shufftok, normalize = True))
        mccs_c1.append(mcc_c1)
        
        test_df['shuffchar_class'] = test_df['scrambled_toehold'].apply(lambda x: pred_class(x, learn_cf))
        y_test_shuffchar = test_df['shuffchar_class']
        mcc_c2 = mcc(y_test_true, y_test_shuffchar)
        c2_scores.append(accuracy_score(y_test_true, y_test_shuffchar, normalize = True))
        mccs_c2.append(mcc_c2)
        
        scores.append((acuf + acur)/2) 
        aucs.append((roc_aucf + roc_aucr)/2)
        mccs.append((mccf + mccr)/2)
        fracs.append(sampling)
        trials.append(trial)
        
        learn_cf.save('toe_cfF')
        learn_cf.save_encoder('toe_cfF_enc')
        
        learn_cr.save('toe_cfR')
        learn_cr.save_encoder('toe_cfR_enc')
        
        learn_cf.destroy()
        learn_cr.destroy()

df_eval = pd.DataFrame()
df_eval['trial'] = trials
df_eval['sampling_fraction'] = fracs
df_eval['model_accuracy'] = scores
df_eval['model_roc_auc'] = aucs
df_eval['matthews_cc'] = mccs
df_eval['c1_accuracy'] = c1_scores
df_eval['c2_accuracy'] = c2_scores
df_eval['c1_mcc'] = mccs_c1
df_eval['c2_mcc'] = mccs_c2
df_eval.to_csv(pltpath+'Results_RandLM_SampONOFF.csv')
print(df_eval)

