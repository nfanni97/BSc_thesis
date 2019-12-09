# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:56:06 2017

@author: Övgü
"""

# -*- coding: utf-8 -*-
"""
Created on Oct 1 17:58 2017

@author: Övgü Özdemir
"""

from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras import backend as K
from sklearn.metrics import confusion_matrix
import scipy.io as spio
import numpy as np
import traceback

def write_matrices(filename,data):
    with open(filename,'w') as outfile:
        #header:
        outfile.write('# Array shape: {0}\n'.format(data.shape))
        #slices:
        for data_slice in data:
            np.savetxt(outfile, data_slice,fmt='%-2d')
            outfile.write('# New slice\n')

inPath = "/home/nagfa5/imaging/source/05/"
outPath = "/home/nagfa5/imaging/results/"
numofclasses = 4*3
num_days = 7
num_test = 240
num_train = 1080
num_iter = 20
batch_size = 16

filename_prefix = "05_results"
filename = outPath+filename_prefix+".csv"
filename_m = outPath+filename_prefix+"_matrices.csv"

test_data = np.load(inPath+"test_data_without_PAA.npy")
train_data = np.load(inPath+"train_data_without_PAA.npy")
train_label = np.load(inPath+"train_label.npy")
test_label = np.load(inPath+"test_label.npy")

size = [test_data.shape[1],test_data.shape[2]]

test_data /= np.max(test_data)
test_data = np.reshape(test_data,(num_test,size[0],size[1],1))
train_data /= np.max(train_data)
train_data = np.reshape(train_data,(num_train,size[0],size[1],1))

result = np.zeros((num_iter,1),dtype=np.double)
matrices = np.zeros((num_iter,numofclasses,numofclasses),dtype=np.int)

try:
    for i in range(num_iter):
        model=Sequential()
        model.add(Conv2D(24, (3,3), input_shape=(size[0], size[1], 1), padding='same',
                          activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Conv2D(24, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(2Dropout(0.2))
        model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.2))
        model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(numofclasses, activation='softmax'))
        #gd = SGD(lr = 0.1, decay = 0.1, momentum = 0.9, nesterov = True)
        model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])
            #print(model.summary())
        model.fit(train_data, train_label, batch_size=batch_size, epochs=50, verbose=0)
        #calculate output on test data
        predictions_test = model.predict(test_data)
        matrices[i,:,:] = confusion_matrix(test_label.argmax(axis=1),predictions_test.argmax(axis=1))
        score, acc = model.evaluate(test_data, test_label, batch_size=batch_size, verbose=0)
        result[i] = acc
        print("test "+str(i)+": "+str(acc))
        K.clear_session()
    np.savetxt(filename,result)
    write_matrices(filename_m,matrices)
except KeyboardInterrupt as k:
    np.savetxt(filename, result)
    write_matrices(filename_m, matrices)
    print('Script was stopped.')
    f = open(outPath+filename_prefix+"stopped",'w+')
    f.close()
except Exception as e:
    print('An exception occured.')
    np.savetxt(filename, result)
    write_matrices(filename_m, matrices)
    f = open(outPath + filename_prefix + "exception",'w+')
    f.write(str(e))
    f.close()
    traceback.print_exc()
