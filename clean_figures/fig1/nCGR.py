## Import Libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from itertools import islice
import re
import math

import pickle
import random
from datacleaner_updated import *
from Bio.SeqUtils import GC
from Bio import motifs

sns.set_context('paper', font_scale = 1.5, rc = {"lines.linewidth": 1.5})

sns.set_style('ticks', {'axes.grid': False, 'grid.linestyle': '', 
                            'font.family':'sans-serif', 
                            'font.sans-serif':'Myriad Pro',
                            'text.color': '0',
                            'xtick.color': '0',
                            'ytick.color': '0'
                           })
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler(feature_range=(-1,1))



# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# define all the CGR functions 
def getNucleotide(x,y):
    if (x<0 and y>0):
        nucleotide = 'A'
    elif (x>0 and y>0):
        nucleotide = 'G'
    elif  (x>0 and y<0):
        nucleotide ='T'
    elif  (x<0 and y<0):
        nucleotide = 'C'
    else:
        nucleotide ='N'

    return nucleotide

def getCGRVertex(x,y):
    Nx = 0;
    Ny = 0;
 
    if (x>0 and y>0):
        Nx = 1
        Ny = 1
    elif (x>0 and y<0):
        Nx = 1
        Ny = -1
    elif  (x<0 and y>0):
        Nx = -1
        Ny = 1
    elif  (x<0 and y<0):
        Nx = -1
        Ny = -1
    else:
        Nx = 0
        Ny = 0
   
    return Nx,Ny

def encodeDNASequence(seq):
    A = [-1, 1]
    T = [1, -1]
    C = [-1, -1]
    G = [1, 1]
    a = 0
    b = 0
    x = [] 
    y = []
    n = len(seq)
 
    if seq[0] == 'A':
        a = int(A[0])
        b = int(A[1])
    elif  seq[0] == 'T':
        a = int(T[0])
        b = int(T[1])
    elif  seq[0] == 'C':
        a = int(C[0])
        b = int(C[1])
    else:
        a = int(G[0])
        b = int(G[1])
   
    x.append(a)
    y.append(b)
   
    for i in range(1,n):
        if seq[i] == 'A':
            a = int(x[i-1]) + int(math.pow(2,i))
            b = int(y[i-1]) + int(math.pow(2,i))
        elif seq[i] == 'T':
            a = int(x[i-1]) - int(math.pow(2,i))
            b = int(y[i-1]) + int(math.pow(2,i))
        elif  seq[i] == 'C':
            a = int(x[i-1]) - int(math.pow(2,i))
            b = int(y[i-1]) - int(math.pow(2,i))
        else:
            a = int(x[i-1]) + int(math.pow(2,i))
            b = int(y[i-1]) - int(math.pow(2,i))
   
        x.append(a)
        y.append(b)
  
    x_n = int(x[n-1])
    y_n = int(y[n-1])

    return x_n,y_n, n

def returnDNAencoded(seq):
    A = [-1, 1]
    T = [1, -1]
    C = [-1, -1]
    G = [1, 1]
    a = 0
    b = 0
    x = [] 
    y = []
    n = len(seq)
 
    if seq[0] == 'A':
        a = int(A[0])
        b = int(A[1])
    elif  seq[0] == 'T':
        a = int(T[0])
        b = int(T[1])
    elif  seq[0] == 'C':
        a = int(C[0])
        b = int(C[1])
    else:
        a = int(G[0])
        b = int(G[1])
   
    x.append(a)
    y.append(b)

   
    for i in range(1,n):
        if seq[i] == 'A':
            a = int(x[i-1]) + int(math.pow(2,i))
            b = int(y[i-1]) + int(math.pow(2,i))
        elif seq[i] == 'T':
            a = int(x[i-1]) - int(math.pow(2,i))
            b = int(y[i-1]) + int(math.pow(2,i))
        elif  seq[i] == 'C':
            a = int(x[i-1]) - int(math.pow(2,i))
            b = int(y[i-1]) - int(math.pow(2,i))
        else:
            a = int(x[i-1]) + int(math.pow(2,i))
            b = int(y[i-1]) - int(math.pow(2,i))
   
    x.append(a)
    y.append(b)
  
    x_n = int(x[n-1])
    y_n = int(y[n-1])
   
    return x, y

def decodeDNASequence(x_n,y_n,n):
    x = [0] * n 
    y = [0] * n
    x[n-1] = x_n 
    y[n-1] = y_n
 
    seq=[]
    for i in range(n-1,-1,-1): 
        nt = getNucleotide(x[i],y[i])
        seq.insert(i-n+1,nt)
        Nx, Ny = getCGRVertex(x[i],y[i]) 
        x[i-1] = int(x[i]) - int(math.pow(2,i)*Nx)
        y[i-1] = int(y[i]) - int(math.pow(2,i)*Ny)
    else:  
        DNASeq = ''.join(seq) 
    return DNASeq

def scale_array(dat, out_range=(-1, 1)):
    domain = [np.min(dat, axis=0), np.max(dat, axis=0)]

    def interp(x):
        return out_range[0] * (1.0 - x) + out_range[1] * x

    def uninterp(x):
        b = 0
        if (domain[1] - domain[0]) != 0:
            b = domain[1] - domain[0]
        else:
            b =  1.0 / domain[1]
        return (x - domain[0]) / b

    return interp(uninterp(dat))


def rand_seq(seq_length):
    """
    This function generates a random DNA sequence of desired length
    """
    return ''.join(random.choice('ATCG') for _ in range(seq_length))

def clean_sequence(seq, max_len):
    """
    This function uses the data-cleaner class to clean up sequences
    """
    # initialize datacleaner class
    fs_init = Sequence(seq, sequence_type = 'nucleic_acid')
    seq_clean = fs_init.standardized(max_len)
    return seq_clean

def get_2Dcgr_coords(seqs):
    """
    This function returns the 2D normalized CGR representation of a list of DNA sequences
    """
    
    xigcr = [] # final x
    yigcr = [] # final y

    # get the 2D iCGR coordinates
    for i in range(0, len(seqs)):
        x, y, _ = encodeDNASequence(seqs[i])
        xigcr.append(x)
        yigcr.append(y)

    # convert to np array
    xigcr = np.array(xigcr).reshape(-1,1)
    yigcr = np.array(yigcr).reshape(-1,1)

    # scale between -1 to 1
    mm_scaler = MinMaxScaler(feature_range = (-1,1))
    x_norm = mm_scaler.fit_transform(xigcr).reshape(-1,)
    y_norm = mm_scaler.fit_transform(yigcr).reshape(-1,)
    
    return x_norm, y_norm

def plot_2Dcgr(x_norm, y_norm, title, axis):
    """
    This function plots the normed 2D CGR coordinates.
    """
    # plot the good guides
    sns.kdeplot(data = x_norm, 
                data2 = y_norm,
                shade = True,
                cmap = 'Greys',
                cbar = True,
                shade_lowest = True,
                ax = axis
                 )

    axis.set_xlabel('$x_{n = N}$')
    axis.set_xlim([-1.25, 1.25])
    axis.set_ylabel('$y_{n = N}$')
    axis.set_ylim([-1.25, 1.25])
    axis.set_title(title)