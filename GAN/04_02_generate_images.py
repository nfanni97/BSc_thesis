# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 11:17:18 2014
Modified on Wed Jan 27 15:36:00 2016
@author: Stephen Wang
"""

#from serie2QMlib import *
import numpy as np
import pandas as pd
import sys

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
    q = pd.qcut(list(set(series)), Q)
    dic = dict(zip(set(series), q.labels))
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
    return np.array(MSM), label, q.levels

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
def paaMarkovMatrix(paalist,level):
    paaindex = []
    for each in paalist:    
        for k in range(len(level)):
            lower = float(level[k][1:-1].split(',')[0])
            upper = float(level[k][1:-1].split(',')[-1])
            if each >=lower and each <= upper:
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

#save output
def saveOutput(mat,name):
    print('saving...')
    np.save(name,mat)
    print('saved')


#################################
###Define the parameters here####
#################################


path = '/home/nagfa5/GAN/04_groups/'
regions = ['middle','north','south']
datafiles = ['01_data_'+region+'.npy' for region in regions] # Data file name
print(datafiles)
size = [50]  # PAA size
GAF_type = 'GASF' # GAF type: GASF, GADF
rescale_type = 'Zero' # Rescale the data into [0,1] or [-1,1]: Zero, Minusone

for datafile in datafiles:
    fn = datafile
    for s in size:  
        print('read file: {}, size: {}, GAF type: {}, rescale_type: {}'.format(datafile, s, GAF_type, rescale_type))
        raw = np.load(path+datafile).tolist()
        
        print('format data')
        image = []
        paaimage = []
        patchimage = []
        matmatrix = []
        fullmatrix = []
        for each in raw:
            if rescale_type == 'Zero':
                std_data = rescale(each)
            elif rescale_type == 'Minusone':
                std_data = rescaleminus(each)
            else:
                sys.exit('Unknown rescaling type!')
            paalistcos = paa(std_data,s,None)

            datacos = np.array(std_data)
            datasin = np.sqrt(1-np.array(std_data)**2)

            paalistcos = np.array(paalistcos)
            paalistsin = np.sqrt(1-paalistcos**2)
            
            datacos = np.matrix(datacos)
            datasin = np.matrix(datasin)            
            
            paalistcos = np.matrix(paalistcos)
            paalistsin = np.matrix(paalistsin)            
            if GAF_type == 'GASF':
                paamatrix = paalistcos.T*paalistcos-paalistsin.T*paalistsin
                matrix = np.array(datacos.T*datacos-datasin.T*datasin)
            elif GAF_type == 'GADF':
                paamatrix = paalistsin.T*paalistcos-paalistcos.T*paalistsin
                matrix = np.array(datasin.T*datacos - datacos.T*datasin)
            else:
                sys.exit('Unknown GAF type!')
            paamatrix = np.array(paamatrix)
            image.append(matrix)
            paaimage.append(np.array(paamatrix))
            matmatrix.append(paamatrix)
            fullmatrix.append(matrix)
    
        matmatrix = np.array(matmatrix)
        fullmatrix = np.array(fullmatrix)
        image = np.asarray(image)
        paaimage = np.asarray(paaimage)
        
        saveOutput(matmatrix,datafile+'_with_PAA')
        saveOutput(fullmatrix,datafile+'_without_PAA')

        print(matmatrix.shape)
        print(fullmatrix.shape)
