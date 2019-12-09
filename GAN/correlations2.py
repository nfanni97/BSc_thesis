# same as correlation.py but mean is only computed after all correlation values are calculated
import numpy as np

inPath = '/home/nagfa5/GAN/04_groups/'
outPath = '/home/nagfa5/GAN/04_analysis/'
regions = ['middle','north','south']

def rescale(data):
	data_min = np.amin(data,axis=1,keepdims=True)
	data_max = np.amax(data,axis=1,keepdims=True)
	return (data - data_min) / (data_max - data_min)

def correlation_value(data_1,data_2):
	corrs = np.empty(shape=(data_1.shape[0],data_2.shape[0]))
	for i in range(data_1.shape[0]):
		for j in range(data_2.shape[0]):
			corrs[i,j] = np.corrcoef(data_1[i,:],data_2[j,:])[1,0]
	return np.mean(corrs)

corr_1 = np.empty(shape=(3))
original = np.empty(shape=(3,5928,168))
generated = np.empty(shape=(3,100,168))

for i in range(3):
	print(regions[i])
	# 0. load and rescale
	original[i,:,:] = rescale(np.load(inPath+'01_data_'+regions[i]+'.npy'))
	generated[i,:,:] = rescale(np.load(outPath+'generated_time_'+regions[i]+'.npy'))

	# 1. correlation between original and generated images
	corr_1[i] = correlation_value(original[i,:,:],generated[i,:,:])
	print(corr_1[i])
np.save(outPath+'corr_1_no_mean',corr_1)
