# -*- coding: utf-8 -*-
"""
Created on Fri Aug 15 22:18:57 2014
Modified on Wed Jan 27 15:36:00 2016
@author: Stephen Wang
"""

from serie2QMlib import *
import numpy as np
import pandas as pd

#Define sliding window
def window_time_series(series, n, step = 1):
#    print "in window_time_series",series
    if step < 1.0:
        step = max(int(step * n), 1)
    return [series[i:i+n] for i in range(0, len(series) - n + 1, step)]

#PAA function
def paa(series, now, opw):
    if now == None:
        now = int(len(series) / opw)
    if opw == None:
        opw = int(len(series) / now)
    return [sum(series[i * opw : (i + 1) * opw]) / float(opw) for i in range(now)]

def standardize(serie):
    dev = np.sqrt(np.var(serie))
    mean = np.mean(serie)
    return [(each-mean)/dev for each in serie]

#Rescale data into [0,1]
def rescale(serie):
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval-minval)
    return [(each-minval)/gap for each in serie]
#Rescale data into [-1,1]
def rescaleminus(serie):
    maxval = max(serie)
    minval = min(serie)
    gap = float(maxval-minval)
    return [(each-minval)/gap*2-1 for each in serie]

#Generate quantile bins
def QMeq(series, Q):
    q_labels = pd.qcut(list(series), Q, labels=False)
    q_levels = pd.qcut(list(series), Q, labels=None)
    dic = dict(zip(series, q_labels))
    MSM = np.zeros([Q,Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0, len(label)-1):
        MSM[label[i]][label[i+1]] += 1
    for i in range(Q):
        if sum(MSM[i][:]) == 0:
            continue
        MSM[i][:] = MSM[i][:]/sum(MSM[i][:])
    return np.array(MSM), q_labels, q_levels.categories.get_values().tolist()

#Generate quantile bins when equal values exist in the array (slower than QMeq)
def QVeq(series, Q):
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.labels))
    qv = np.zeros([1,Q])
    label = []
    for each in series:
        label.append(dic[each])
    for i in range(0,len(label)):
        qv[0][label[i]] += 1.0
    return np.array(qv[0][:]/sum(qv[0][:])), label

#Generate Markov Matrix given a spesicif number of quantile bins
def paaMarkovMatrix(paalist,levels):
    paaindex = []
    for each in paalist:
        for level in levels:
            if each >=level.left and each <= level.right:
                paaindex.append(k)
    return paaindex

#return the max value instead of mean value in PAAs
def maxsample(mat, s):
    retval = []
    x, y, z = mat.shape
    l = np.int(np.floor(y/float(s)))
    for each in mat:
        block = []
        for i in range(s):
            block.append([np.max(each[i*l:(i+1)*l,j*l:(j+1)*l]) for j in range(s)])
        retval.append(np.asarray(block))
    return np.asarray(retval)

#Pickle the data and save in the pkl file
def pickledata(mat, label, train, name):
    print('..pickling data:'.format(name))
    traintp = (mat[:train], label[:train])
    testtp = (mat[train:], label[train:])
    with open(name+'.pkl', 'wb') as f:
        pickletp = [traintp, testtp]
        pickle.dump(pickletp, f, protocol=pickle.HIGHEST_PROTOCOL)

def pickle3data(mat, label, train, name):
    print('..pickling data:'.format(name))
    traintp = (mat[:train], label[:train])
    validtp = (mat[:train], label[:train])
    testtp = (mat[train:], label[train:])
    with open (name+'.pkl', 'wb') as f:
        pickletp = [traintp, validtp, testtp]
        pickle.dump(pickletp, f, protocol=pickle.HIGHEST_PROTOCOL)

#save output
def saveOutput(mat,name):
    print('saving...')
    np.save(name,mat)
    print('saved')

#################################
###Define the parameters here####
#################################

path = '/home/nagfa5/imaging/source/09/'
datafiles = [path+'test_data',path+'train_data'] # Data fine name
size = [168]  # PAA size
quantile = [16] # Quantile size
reduction_type = 'patch' # Reduce the image size using: full, patch, paa


for datafile in datafiles:
    fn = datafile
    for s in size:  
        for Q in quantile:
            print('read file: {}, size: {}, reduction_type: {}'.format(datafile, s, reduction_type))
            raw = open(fn).readlines()
            raw = [list(map(float, each.strip().split())) for each in raw]
            length = len(raw[0])
            
            print('format data')
            paaimage = []
            paamatrix = []
            patchimage = []
            patchmatrix = []
            fullimage = []
            fullmatrix = []
            for each in raw:
                std_data = each
                paalist = paa(std_data,s,None)
                
                ############### Markov Matrix #######################
                mat, matindex, level = QMeq(std_data, Q)
                paamatindex = paaMarkovMatrix(paalist, level)
                column = []
                paacolumn = []
                for p in range(len(std_data)):
                    for q in range(len(std_data)):
                        column.append(mat[matindex[p]][matindex[(q)]])
                        
                for p in range(s):
                    for q in range(s):
                        paacolumn.append(mat[paamatindex[p]][paamatindex[(q)]])
                        
                column = np.array(column)
                columnmatrix = column.reshape(len(std_data),len(std_data))
                fullmatrix.append(column)
                paacolumn = np.array(paacolumn)
                paamatrix.append(paacolumn)
                
                fullimage.append(column.reshape(len(std_data),len(std_data)))
                paaimage.append(paacolumn.reshape(s,s))
                
                batch = int(len(std_data)/s)
                patch = []
                for p in range(s):
                    for q in range(s):
                        patch.append(np.mean(columnmatrix[p*batch:(p+1)*batch,q*batch:(q+1)*batch]))
                patchimage.append(np.array(patch).reshape(s,s))
                patchmatrix.append(np.array(patch))
 
            paaimage = np.asarray(paaimage)
            paamatrix = np.asarray(paamatrix)
            patchimage = np.asarray(patchimage)
            patchmatrix = np.asarray(patchmatrix)
            fullimage = np.asarray(fullimage)
            fullmatrix = np.asarray(fullmatrix)
            
            if reduction_type == 'patch':
                savematrix = patchmatrix
            elif reduction_type == 'paa':
                savematrix = paamatrix
            else:
                savematrix = fullmatrix
                
            datafilename = datafile +'_'+reduction_type+'_PAA_'+str(s)+'_Q_'+str(Q)+'_MTF'
            print(fullmatrix.shape)
            print(paamatrix.shape)
            print(patchmatrix.shape)
            saveOutput(fullmatrix,datafilename)
            saveOutput(paamatrix,datafilename)
            saveOutput(patchmatrix,datafilename)