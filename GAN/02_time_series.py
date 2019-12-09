import numpy as np

inPath = '/home/nagfa5/GAN/04_groups/'
outPath = '/home/nagfa5/GAN/04_analysis/'

def save_data(data,name):
	np.save(outPath+name,data)

#load data
def load_data(name,from_in=True):
	if from_in:
		return np.load(inPath+name+'.npy')
	else:
		return np.load(outPath+name+'.npy')

middle_original = load_data('01_data_middle')
north_original = load_data('01_data_north')
south_original = load_data('01_data_south')

middle_gen = load_data('generated/middle/2000')
north_gen = load_data('generated/north/2000')
south_gen = load_data('generated/south/2000')

#rescale original values to [0,1]
def rescale(data,min=0,max=1):
	data_min = np.amin(data,axis=1,keepdims=True)
	data_max = np.amax(data,axis=1,keepdims=True)
	data_normed = (data - data_min) / (data_max - data_min)
	return data_normed

middle_rescaled = rescale(middle_original)
north_rescaled = rescale(north_original)
south_rescaled = rescale(south_original)

#compute mean of original values
middle_mean = np.mean(middle_rescaled,axis=0,keepdims=True)
north_mean = np.mean(north_rescaled,axis=0,keepdims=True)
south_mean = np.mean(south_rescaled,axis=0,keepdims=True)

#convert back to time series
def convert_to_time_series(data,GASF=True):
	time_series = np.zeros((data.shape[0],data.shape[1]))
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			time_series[i,j] = np.cos(np.arccos(data[i,j,j]) / 2)
	return time_series

middle_series = convert_to_time_series(middle_gen)
north_series = convert_to_time_series(north_gen)
south_series = convert_to_time_series(south_gen)

#compute correlation matrix
def correlation_matrix(mean_data,gen_data):
	corr = np.empty((gen_data.shape[0],2,2))
	for i in range(gen_data.shape[0]):
		corr[i,:] = np.corrcoef(mean_data[0,:],gen_data[i,:])
	return corr

middle_corr = correlation_matrix(middle_mean,middle_series)
north_corr = correlation_matrix(north_mean,north_series)
south_corr = correlation_matrix(south_mean,south_series)

save_data(middle_corr,'middle_corr')
save_data(north_corr,'north_corr')
save_data(south_corr,'south_corr')
