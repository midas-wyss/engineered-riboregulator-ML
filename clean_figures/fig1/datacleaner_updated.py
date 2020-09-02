import functools
import itertools
import math
import os
from os.path import splitext
import pickle
import re
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from Bio import Seq, SeqIO
from Bio.Seq import Seq
from Bio.Alphabet import generic_dna, generic_rna
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import FeatureLocation, CompoundLocation

from scipy import stats, interp

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import average_precision_score as apscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error 
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import r2_score as r2
from scipy.stats import spearmanr as spr
from sklearn.metrics import cohen_kappa_score as cks
from sklearn.metrics import balanced_accuracy_score as bacc
from sklearn.metrics import make_scorer
from scipy.stats import ks_2samp as ks2

# Library containing list of glycoletters from SweetTalk
with open('/home/pradeep/Documents/TPOT/main_classes/glycoletter_lib.pkl', 'rb') as file:
    glycoletter_lib = pickle.load(file)
    
def small_glymotif_find(s):
    """ Breaks down glycans into list of monomers. Assumes
        that user provides glycans in a consistent universal 
        notation. Input is a glycan string
    """
    b = s.split('(')
    b = [k.split(')') for k in b]
    b = [item for sublist in b for item in sublist]
    b = [k.strip('[') for k in b]
    b = [k.strip(']') for k in b]
    b = [k.replace('[', '') for k in b]
    b = [k.replace(']', '') for k in b]
    b = '*'.join(b)
    return b

def process_glycans(glycan_list):
    """ Wrapper function to process list of glycans into glycoletters.
        Input is a list of glycans. Output is a list of glycowords
    """
    glycan_motifs = [small_glymotif_find(k) for k in glycan_list]
    glycan_motifs = [k.split('*')for k in glycan_motifs]
    return glycan_motifs

# Custom InputError to raise when input data is invalid
class Error(Exception):
    """ Base class for exceptions in this module."""
    pass

class InputError(Error):
    """ Exception raised for errors in the input.

    Attributes:
        expression: input expression in which the error occurred
        message: explanation of the error
    """

    def __init__(self, message):
        self.message = message

# Constants related to the allowed alphabets: nucleotide or protein sequences
NUCLEIC_ACID = 'nucleic_acid'
PROTEIN = 'protein'
GLYCAN = 'glycan'

# this is the dictionary of allowed letters/words for each biological sequence-type
ALPHABETS = {
    NUCLEIC_ACID: list('ATCG'),
    PROTEIN: list('ARNDCEQGHILKMFPSTWYVUO'),
    GLYCAN: glycoletter_lib
}

# this is the dictionary of letters to be used when gaps are found within the sequence
GAP_LETTERS = {
    NUCLEIC_ACID: 'N',
    PROTEIN: 'X',
    GLYCAN: 'X'
}

# this is the dictionary of letters to be used when sequences are not of the same length
PAD_LETTERS = {
    NUCLEIC_ACID: 'n', # use lower-case n to indicate padding
    PROTEIN: 'x', # use lower-case x to indicate padding
    GLYCAN: 'Y' # use captial Y to indicate padding
}

# this is the dictionary of allowed substitutions to convert gaps and non-standard letters into the standard alphabet
SUBSTITUTIONS = {
    NUCLEIC_ACID: {
        'R' : 'AG',
        'U' : 'T',
        'Y' : 'CT',
        'S' : 'GC',
        'W' : 'AT',
        'K' : 'GT',
        'M' : 'AC',
        'B' : 'CGT',
        'D' : 'AGT',
        'H' : 'ACT',
        'V' : 'ACG',
        'N' : ALPHABETS[NUCLEIC_ACID]
    },
    PROTEIN: {
        'B' : 'RN',
        'Z' : 'EQ',
        'J' : 'LI',
        'U' : 'U', # Selenocysteine
        'O' : 'O', # Pyrrolysine
        'X' : ALPHABETS[PROTEIN]
    },
    GLYCAN: {
        'X' : ALPHABETS[GLYCAN]
    }
}

AUGMENTATION_TYPES = ['none', 'complement', 'reverse_complement']
TARGET_TYPES = ['string', 'integer', 'continuous']

class Alphabet(object):

    def __init__(self, sequence_type):
        """ Object for representing a nucleic acid or protein or glycan alphabet

        Attributes:
            sequence_type (chr): one of the allowed sequence types from ALPHABETS
        """
        if sequence_type in ALPHABETS:
            self.sequence_type = sequence_type
            self.alphabet = ALPHABETS[sequence_type]
            self.substitutes = SUBSTITUTIONS[sequence_type]
            self.gap_letter = GAP_LETTERS[sequence_type]
            self.pad_letter = PAD_LETTERS[sequence_type]
        else:
            raise InputError('sequence_type "{0}" is invalid. Valid alphabet types are: {1}'.format(sequence_type, ', '.join(ALPHABETS)))

    def standard_letters(self):
        return ''.join(sorted(self.alphabet))
    
    def extended_letters(self):
        return ''.join(sorted(self.alphabet + list(self.pad_letter)))

    def all_letters(self):
        return ''.join(sorted(self.alphabet + list(self.substitutes) + list(self.gap_letter) + list(self.pad_letter)))

    def generate_random_sequence(self, length):
        """ Generate a random sequence drawn from an alphabet.

        Arguments:
            length (int): length of random sequence to be generated
            sequence_type (chr): one of the allowed ALPHABETS (nucleic_acid, protein, glycan)
        """
        try:
            length = int(length)
        except ValueError:
            raise InputError('length argument: "{}" is not an integer.'.format(length))

        return ''.join(random.choice(self.alphabet) for _ in range(length))


class Sequence(object):

    def __init__(self, sequence, sequence_type):
        """ Object for representing nucleic acid or protein or glycan sequences

        Attributes:
            sequence (chr): the sequence itself, e.g 'ATCGAGT' or 'EQGHILKMFP' or 'Gal(b1-2)[Gal(b1-3)]Gal(b1-6)[Galf(b1-2)Galf(b1-4)]GlcNAc'
            sequence_type (chr): one of the allowed sequence types from ALPHABETS
        """
        if sequence_type in ALPHABETS:
            self.sequence = sequence
            self.sequence_type = sequence_type
            self.alphabet = Alphabet(sequence_type)
            self.sequence_standardized = None
        else:
            raise InputError('sequence_type "{0}" is invalid. Valid sequence_type are: {1}'.format(sequence_type, ', '.join(ALPHABETS)))

    def fill(self):
        """ Replace gaps ("-") with the corresponding gap letter.
            Not done for glycans.
        """
        if self.sequence_type == 'glycan':
            pass
        else:
            # takes the upper-case for all nucleic_acid and protein alphabets
            seq = self.sequence.upper()
            filled_seq = seq.replace('-', self.alphabet.gap_letter)
            self.sequence = filled_seq

    def pad(self, length):
        """ Pad sequence to desired length using the appropriate gap letter """
        if self.sequence_type == 'glycan':
            # since gaps are excluded and not filled
            seq = self.sequence
        else:
            seq = self.sequence.upper()
            
        len_seq = len(seq)
        delta = length - len_seq
        
        if delta >= 0:
            # pads using the gap letter token
            padding = ''.join(self.alphabet.pad_letter for _ in range(delta))
            seq += padding
            self.sequence = seq
        else:
            raise InputError('Delta ({}) is less than sequence length'.format(delta))

    def standardized(self, length = None):
        """ Return the fully standardized sequence

        Fill gaps, add padding, map non-standard letters and ensure all letters are valid
        """
        # fill gaps
        self.fill()
        
        # pad to same length
        if length:
            self.pad(length)
            
        # substitute non-standard letters 
        seq = self.sequence
        substitutes = self.alphabet.substitutes
        
        for old, new in substitutes.items():
            
            if self.sequence_type == 'glycan':
                seq = [letter if letter != old else random.choice(new) for letter in seq]
            else:
                seq = seq.replace(old, random.choice(new))
        
        # check to make sure that new letters are within defined alphabet
        valid_letters = self.alphabet.all_letters()
        valid = [x in valid_letters for x in seq]

        if all(valid): 
            self.sequence = seq
            return seq
        else:
            raise InputError('Unknown letter(s) "{0}" found in sequence'.format(', '.join([seq[i] for i, x in enumerate(valid) if not x])))

    def scrambled(self):
        """ Return the scrambled sequence

        Ensures that the scrambled output returned has the same letter frequency
        but in a different order than the original sequence. Negative control.
        """
        if self.sequence_standardized:
            original_sequence = self.sequence_standardized
        else:
            original_sequence = self.sequence

        scrambled_sequence = original_sequence

        if len(set(original_sequence)) == 1:
            return scrambled_sequence
        else:
            while scrambled_sequence == original_sequence:
                chars = list(self.sequence)
                random.shuffle(chars)
                if self.sequence_type == 'glycan':
                    scrambled_sequence = chars
                else:
                    scrambled_sequence = ''.join(chars)
        return scrambled_sequence

    def augmented(self, augmentation_type):
        """ Return the augmented (complemented/reverse complemented) sequence. Applicable for nucleic_acid only! """
        if augmentation_type in AUGMENTATION_TYPES:
            if (self.sequence_type != NUCLEIC_ACID) & (augmentation_type != 'none'):
                print("Augmentation is only possible for sequence_type='nucleic_acid'. Setting augmentation_type to 'none'.'")
                self.augmentation_type = 'none'
            else:
                self.augmentation_type = augmentation_type

            if self.augmentation_type == 'complement':
                return Seq(self.sequence).complement().__str__()
            elif self.augmentation_type == 'reverse_complement':
                return Seq(self.sequence).reverse_complement().__str__()
            else:
                return None

        else:
            raise InputError('augmentation_type "{0}" is invalid. Valid values are: {1}'.format(augmentation_type, ', '.join(AUGMENTATIONS)))


def read_data(data_path):
    """ Read input data in csv, .xls, or .xlxs format """
    _, ext = splitext(data_path)
    if ext == '.csv':
        try:
            data = pd.read_csv(data_path)
        except:
            raise
    elif (ext == '.xls') or (ext == '.xlsx'):
        try:
            data = pd.read_excel(data_path)
        except:
            raise
    else:
        raise InputError('Unsupported data format. Please convert to csv, xls, or xlsx')
    return data

# helper functions to automatically determine the appropriate threshold from continuous data.
# currently works only for sequence_type = 'nucleic_acid' and 'protein'

def partition_data(df_processed, threshold):
    """ Partitions the processed dataframe into two classes provided that the target is continuous.
        The input must already be cleaned using the data-cleaner.
    """
    # split the processed df into two classes with above and below the specified threshold
    df_above = df_processed[df_processed['target'] > threshold]
    df_above['above_thresh'] = 1
    df_below = df_processed[df_processed['target'] <= threshold]
    df_below['above_thresh'] = 0
    
    # concatenate back together
    df_ml = pd.concat([df_above, df_below], axis = 0)
    df_ml = df_ml.sample(frac = 1).reindex()
    X = df_ml['sequence']
    X_control = df_ml['scrambled_sequence']
    y = df_ml['above_thresh']
    
    return X, X_control, y

def split_transform_data(X, X_control, y):
    """Splits the processed data into training, testing, and controls."""
    # split the data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, stratify = y)
    # fit the tf-idf vectorizer on the training set
    # here we are using a character-level model consisting of unigrams and bigrams as the simplest model. 
    tfidf_vec = TfidfVectorizer(lowercase = False, ngram_range = (1,2), analyzer = 'char')
    x_train_idf = tfidf_vec.fit_transform(x_train)
    # transform the test set similarly
    x_test_idf = tfidf_vec.transform(x_test)
    # transform the control set similarly
    x_control_idf = tfidf_vec.transform(X_control)
    
    return x_train_idf, y_train, x_test_idf, y_test, x_control_idf

def evaluate_partitions(keep_bin_edges, df_processed):
    """ This function evaluates a lightweight classifier according to the thresholds.
        Inputs are a list of bin-edges for the continuous target and the processed df.
    """
    # initialize the empty lists 
    accs = []
    aucs = []
    mccs = []
    apcs = []

    accs_control = []
    aucs_control = []
    mccs_control = []
    apcs_control = []

    threshs = []
    bin_pct = []

    # starting data percentile
    pct = 0.0
    # binning parameters fixed - DO NOT CHANGE
    num_bins = 10
    num_trials = 10
    # sweep through all bin edges
    for bin_edge in keep_bin_edges:
        
        threshold = bin_edge
        # obtain the X,y matrices
        X, X_control, y = partition_data(df_processed, threshold)
        # starting data percentile
        pct += 1/num_bins
        for trial in range(num_trials):
            # get the training, testing, and control data-sets
            x_train_idf, y_train, x_test_idf, y_test, x_control_idf = split_transform_data(X, X_control, y)
            # fit the classifier
            clf = ComplementNB(alpha = 0.1, class_prior = None, fit_prior = True, norm = False)
            clf.fit(x_train_idf, y_train)
        
            # evaluate on test and control sets
            accs.append(clf.score(x_test_idf, y_test)) 
            accs_control.append(clf.score(x_control_idf, y))
        
            y_pred = clf.predict(x_test_idf)
            y_pred_cont = clf.predict(x_control_idf)
        
            mccs.append(mcc(y_test, y_pred))
            mccs_control.append(mcc(y, y_pred_cont))
        
            y_proba = clf.predict_proba(x_test_idf)
            y_cont_proba = clf.predict_proba(x_control_idf)
        
            aucs.append(roc_auc_score(y_test, y_proba[:,1]))
            aucs_control.append(roc_auc_score(y, y_cont_proba[:,1]))
        
            apcs.append(apscore(y_test, y_proba[:,1]))
            apcs_control.append(apscore(y, y_cont_proba[:,1]))
        
            threshs.append(threshold)
            bin_pct.append(pct)

    # populate into a df for downstream analysis
    df_eval = pd.DataFrame()
    df_eval['data percentile'] = bin_pct # data percentile 
    df_eval['threshold'] = threshs # bin edge
    df_eval['test accuracy'] = accs # accuracy
    df_eval['test mcc'] = mccs # matthews correlation coefficient
    df_eval['test auc'] = aucs # roc-auc
    df_eval['test ap'] = apcs # average precision
    df_eval['control accuracy'] = accs_control
    df_eval['control mcc'] = mccs_control
    df_eval['control auc'] = aucs_control
    df_eval['control ap'] = apcs_control
    
    return df_eval

def compute_tests(df_eval):
    """ Takes the result of the data splitting as a function of various percentiles and computes
        significance values between the test and control sets. 
    """
    # df of all unique data-percentiles within the data
    dps = df_eval['data percentile'].value_counts()
    # list of all unique data percentiles
    dps_list = dps.index.values.tolist()

    # empty list for p-values
    acc_p = []
    mcc_p = []
    auc_p = []
    aps_p = []

    pcts = []
    threshs = []
    gms = []
    
    # iterate over all unique data percentiles
    for pct in dps_list:
        # slice out all the trials for a specific data percentile
        df_temp = df_eval[df_eval['data percentile'] == pct]
        # get the corresponding bin threshold
        th = df_temp['threshold'].iloc[0]
        # conduct the KS two-tailed test on the test and control distributions and get p-values
        acc_temp = ks2(df_temp['test accuracy'], df_temp['control accuracy'])[1]
        mcc_temp = ks2(df_temp['test mcc'], df_temp['control mcc'])[1]
        auc_temp = ks2(df_temp['test auc'], df_temp['control auc'])[1]
        aps_temp = ks2(df_temp['test ap'], df_temp['control ap'])[1]
        
        # concatenate into a list
        a = [mcc_temp, auc_temp, aps_temp]
        # compute the geometric mean of all the p-values
        gm = stats.mstats.gmean(a)
    
        acc_p.append(acc_temp)
        mcc_p.append(mcc_temp)
        auc_p.append(auc_temp)
        aps_p.append(aps_temp)
        gms.append(gm)
    
        pcts.append(pct)
        threshs.append(th)
    
    df_stats = pd.DataFrame()
    df_stats['threshold'] = threshs
    df_stats['data percentile'] = pcts
    df_stats['accuracy pstat'] = acc_p
    df_stats['mcc pstat'] = mcc_p
    df_stats['auc pstat'] = auc_p
    df_stats['ap pstat'] = aps_p
    df_stats['geo mean'] = gms
    df_stats = df_stats.sort_values(by = ['data percentile'])
    
    return df_stats

def repeat_auto_partitioning(keep_bin_edges, df_processed):
    """ Repeats the training of a Naive-Bayes Classifier on the Sequence according to a preset threshold. 
        Returns the results of all calculations as well as the averaged df with a suggestion for the auto-binning.
    """
    fts = []
    dps = []
    gms = []
    
    col_names = ['threshold', 'data percentile','accuracy pstat','mcc pstat','auc pstat','ap pstat','geo mean']
    df_sims = pd.DataFrame(columns = col_names)
    
    # hard-coded: do not change. 
    num_sims = 5
    # repeatedly sample training and testing
    for sim in range(num_sims):
        df_eval = evaluate_partitions(keep_bin_edges, df_processed)
        df_stats = compute_tests(df_eval)
        df_sims = pd.concat([df_sims, df_stats], axis = 0)
    
    # compute the average across all simulations
    pcts = df_sims['data percentile'].value_counts().index.tolist()
    avg_gms = []
    dps = []
    fts = []
    for pct in pcts:
        dps.append(pct)
        tmp = df_sims[df_sims['data percentile'] == pct]
        fts.append(tmp['threshold'].iloc[0])
        avg_gms.append(tmp['geo mean'].mean())
    
    df_avg = pd.DataFrame()
    df_avg['threshold'] = fts
    df_avg['data percentile'] = dps
    df_avg['geo mean'] = avg_gms
    df_avg = df_avg.sort_values(by = ['data percentile'])
    
    # find the scaled target threshold corresponding to the minimum p-value.
    th = df_avg[df_avg['geo mean'] == df_avg['geo mean'].min()]['threshold'].values.tolist()
    th = th[0]
    
    return df_sims, df_avg, th

class DataCleaner(object):

    def __init__(self, data_path, seq_column, target_column, sequence_type,
        target_type, classify, bin_threshold='median', augmentation_type='none'):
        """ Object for transforming input data into a cleaned dataframe

        Attributes:
            data_path (chr): path to .csv or .xlsx file
            seq_column (chr): name of column containing sequences
            target_column (chr): name of column containing target values
            sequence_type (chr): ['nucleic_acid', 'protein', or 'glycan']
            target_type (chr): ['string', 'integer', 'continuous']
            classify (bool): whether to classify continuous data
            bin_threshold: float or 'median' or 'auto'
            augmentation_type (chr): ['none', 'complement', 'reverse_complement']
        """

        # Validate input data, seq_column, and target_column
        try:
            input_data = read_data(data_path)
            self.input_data = input_data
        except:
            raise
        colnames = list(input_data.columns)
        if seq_column in colnames:
            self.seq_column = seq_column
        else:
            raise InputError('seq_column "{}" not found in input data'.format(seq_column))
        if target_column in colnames:
            self.target_column = target_column
        else:
            raise InputError('target_column "{}" not found in input data'.format(target_column))

        self.sequence_type = sequence_type
        self.target_type = target_type
        self.classify = classify
        self.bin_threshold = bin_threshold
        self.augmentation_type = augmentation_type

        # For the final pd.DataFrame() object
        self.cleaned_data = None
        self.num_classes = None
        self.df_sims = None
        self.th = None

    def preprocessed_target_values(self):
        """ Return preprocessed target values """
        label_encoder = LabelEncoder()
        scaler = RobustScaler(with_centering = True, with_scaling = True) # rescales data around the median and is robust to outliers
        imputer_cont = SimpleImputer(missing_values = np.nan, strategy = 'mean') # for continuous targets
        imputer_categ = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent') # for discrete targets

        y_vals = self.input_data[self.target_column].values.reshape(-1,1) # numpy array of targets used for training

        if self.target_type == 'string':
            filled = imputer_categ.fit_transform(y_vals)
            target_output = label_encoder.fit_transform(filled) # creates a numpy array of integer labels
            num_classes = np.unique(target_output).max() + 1

        elif self.target_type == 'integer':
            filled = imputer_categ.fit_transform(y_vals)
            target_output = filled.astype(int) # ensures that input target array is integers
            num_classes = np.unique(target_output).max() + 1

        elif self.target_type == 'continuous':
            filled = imputer_cont.fit_transform(y_vals)
            target_output = scaler.fit_transform(filled) # scaled data has the same shape as input data distribution
            num_classes = None
        else:
            raise InputError('target_type "{0}" is invalid. Valid values are: {1}'.format(self.target_type, ', '.join(TARGET_TYPES)))
        return target_output, num_classes

    def binned_target_values(self):
        """ Return binned and preprocessed target values

        Bins continuous target data to get labels for classification using the
        user-defined threshold, if provided, or defaults to using the median value.
        """
        label_encoder = LabelEncoder()
        target_output, _ = self.preprocessed_target_values()

        if self.bin_threshold == 'median':
            threshed_y = (target_output > np.median(target_output)) # this is the median of the data 
        else:
            threshed_y = (target_output > self.bin_threshold)

        binned_output = label_encoder.fit_transform(threshed_y) # convert boolean labels to integer labels
        num_classes = np.unique(binned_output).shape[0]
        return binned_output, num_classes

    def clean(self):
        """Read and clean the input data.

        Read the input data, process the column containing sequences, and process
        the column containing the targets
        """
        # Read in all the sequences and create a duplicate for the scrambled sequence control
        data = self.input_data
        if self.sequence_type == 'glycan':
            glycans = process_glycans(data[self.seq_column].values.tolist())
            seqs = [Sequence(seq, self.sequence_type) for seq in glycans]
            duplicates = [Sequence(seq, self.sequence_type) for seq in glycans] 
        else:
            seqs = [Sequence(seq, self.sequence_type) for seq in data[self.seq_column]]
            duplicates = [Sequence(seq, self.sequence_type) for seq in data[self.seq_column]]
            
        max_seq_length = np.array([len(seq.sequence) for seq in seqs]).max()
        
        # Standardize them: fill gaps, map non-standard letters, add padding
        standardized_seqs = [seq.standardized(max_seq_length) for seq in seqs]
        
        # Generate scrambled sequences
        pre_scrambled = [seq.scrambled() for seq in duplicates] # first scramble the existing sequence (not standardized)
        scrambled_seqs = [Sequence(seq, self.sequence_type) for seq in pre_scrambled] # create a scrambled sequence function-list to be standardized
        standardized_scrambled_seqs = [seq.standardized(max_seq_length) for seq in scrambled_seqs]

        # Save cleaned sequences, scrambled sequences, and preprocessed target values
        df_processed = pd.DataFrame()
        df_processed['sequence'] = standardized_seqs
        df_processed['scrambled_sequence'] = standardized_scrambled_seqs
        df_processed['target'] = data[self.target_column]

        # Preprocess target values and bin them if classify = True
        if (self.classify == True) & (self.bin_threshold != 'auto'):
            y_processed, num_classes = self.binned_target_values()
            df_processed['processed_target'] = y_processed
            
        elif (self.classify == True) & (self.bin_threshold == 'auto') & (self.sequence_type != 'glycan'):
            # target needs to be 'continuous' for auto-binning algorithm
            # bin the processed data into 20 bins and extract the bin positions
            _, bin_edges = pd.qcut(df_processed['target'], q = 20, retbins = True)
            # exclude the first and last cuts since these are the min and max of the processed data.
            keep_bin_edges = bin_edges[1:-1]
            # run the auto-thresholding algorithm and get the threshold
            df_sims, df_avg, th = repeat_auto_partitioning(keep_bin_edges, df_processed)
            # create classes as-per the determined threshold 
            df_above = df_processed[df_processed['target'] > th]
            df_above['processed_target'] = 1
            df_below = df_processed[df_processed['target'] <= th]
            df_below['processed_target'] = 0
            
            df_processed = pd.concat([df_above, df_below], axis = 0)
            df_processed = df_processed.sample(frac = 1).reindex()
            num_classes = 2
            
            self.df_sims = df_sims
            self.th = th
        else:
            y_processed, num_classes = self.preprocessed_target_values()
            df_processed['processed_target'] = y_processed

        # Check whether we'll be doing augmentation
        aug = seqs[0].augmented(self.augmentation_type)
        if aug:
            # Augment the sequences and shuffle them in with the other data
            df_augment = pd.DataFrame()
            df_augment['sequence'] = [seq.augmented(self.augmentation_type) for seq in seqs]
            df_augment['scrambled_sequence'] = [seq.scrambled() for seq in seqs]
            df_augment['target'] = df_processed['target']
            df_augment['processed_target'] = df_processed['processed_target']
            df_augment.index = data.index
            df_processed = df_processed.append(df_augment, ignore_index=True)
            df_processed = df_processed.sample(frac=1).reindex()

        self.cleaned_data = df_processed
        self.num_classes = num_classes
        

    def plot_target_distributions(self):
        """ Plot cleaned target values as well as the original """
        if self.cleaned_data is None:
            return None

        data = self.cleaned_data
        y_vals = data['target']
        y_scaled = data['processed_target']

        sns.set_style('ticks', {'axes.grid': False,
                                'grid.linestyle': '',
                                'font.family':'sans-serif',
                                'font.sans-serif':'Myriad Pro',
                                'text.color': '0',
                                'xtick.color': '0',
                                'ytick.color': '0'
                               })
        
        if self.target_type != 'string':
            fig, axs = plt.subplots(1,2, figsize = (8,4))
            sns.distplot(y_vals, hist = False, rug = True, ax = axs[0], label = 'Original target')
            axs[0].set_ylabel('Kernel Density Estimate')
            sns.distplot(y_scaled, hist = False, rug = True, ax = axs[1], label = 'Scaled target')
        else:
            fig, axs = plt.subplots(1,1, figsize = (4,4))
            sns.distplot(y_scaled, hist = False, rug = True, ax = axs[0], label = 'Scaled target')
            axs[0].set_ylabel('Kernel Density Estimate')
        return fig, axs
    
    def plot_autobinning_metrics(self):
        """ If continuous target is automatically binned, this function plots the p-values."""
        if self.df_sims is not None:
            fig, axs = plt.subplots(1,2, figsize = (14,5))
            print('----------------------------------------------------------')
            print('Auto Target Threshold:', self.th)
            print('----------------------------------------------------------')
            
            sns.lineplot(x = 'threshold', y = 'geo mean', data = self.df_sims, 
                         dashes = False, markers = True,
                         label = 'Averaged p-value', ax = axs[0])
            
            axs[0].set(xscale = 'log', yscale = 'log')
            axs[0].set(xlabel = 'Target', ylabel = 'KS (Two-tailed) p-value');
            
            sns.lineplot(x = 'threshold', y = 'accuracy pstat', data = self.df_sims, label = 'Model Accuracy', ax = axs[1])
            sns.lineplot(x = 'threshold', y = 'mcc pstat', data = self.df_sims, label = 'Model MCC', ax = axs[1])
            sns.lineplot(x = 'threshold', y = 'auc pstat', data = self.df_sims, label = 'Model ROC-AUC', ax = axs[1])
            sns.lineplot(x = 'threshold', y = 'ap pstat', data = self.df_sims, label = 'Model Average-Precision', ax = axs[1])
            
            axs[1].set(xscale = 'log', yscale = 'log')
            axs[1].set(xlabel = 'Target', ylabel = 'KS (Two-tailed) p-value');
            axs[1].legend(loc='center left', bbox_to_anchor = (1.01, 0.5),
                         ncol = 1, fancybox = False, shadow = False)
        return fig, axs

    def letter_frequency_matrix(self):
        data = self.cleaned_data
        seq_list = data['sequence'].values.tolist()
        alph = Alphabet(self.sequence_type)
        alph_letters = sorted(list(alph.extended_letters()))

        frequency_matrix = np.zeros((len(alph_letters), len(seq_list[0])), dtype = int)
        char2index = {letter: i for (i, letter) in enumerate(alph_letters)}

        for seq in seq_list:
            for index, char in enumerate(seq):
                frequency_matrix[char2index[char]][index] += 1

        return pd.DataFrame(frequency_matrix), char2index

    def plot_character_heatmap(self):
        """ Plot character frequencies as a function of sequence position """

        df_fmat, char2index = self.letter_frequency_matrix()
        df_fmat.index = char2index
        tot_sum = df_fmat.iloc[0].sum()

        df_fmat2 = df_fmat/tot_sum
        df_fmat2.index = char2index

        sns.set_style('ticks', {'axes.grid': False,
                                'grid.linestyle': '',
                                'font.family':'sans-serif',
                                'font.sans-serif':'Myriad Pro',
                                'text.color': '0',
                                'xtick.color': '0',
                                'ytick.color': '0'
                               })

        fig, ax = plt.subplots(figsize = (11.25,1.25))
        sns.heatmap(df_fmat2, linewidths = .1,
                    cmap = sns.color_palette("Blues"),
                    ax = ax, square = True)
        plt.xticks(rotation = 90);
        plt.yticks(rotation = 360);
        ax.set_xlabel('Letter position')

        ticks = list(range(1,df_fmat2.shape[1]+1))
        strs = [str(x) for x in ticks]
        ax.set_xticklabels(strs);

        return fig, ax

    def consensus_sequence(self):
        df_fmat, char2index = self.letter_frequency_matrix()
        index2char = {v:k for (k,v) in char2index.items()}
        max_idx = list(np.argmax(np.array(df_fmat), axis=0))
        consensus = [index2char[i] for i in max_idx]
        return consensus
