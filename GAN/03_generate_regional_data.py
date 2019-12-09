import numpy as np
import sys

def rescale(data):
	min = np.amin(data,axis=1,keepdims=True)
	max = np.amax(data,axis=1,keepdims=True)
	return (data-min)/(max-min)

# data is time series
def spectrum(data):
	ps = np.abs(np.fft.fft(data))**2
	return ps

# data is a float value
def clip(data):
	if data>1:
		return 1
	elif data<0:
		return 0
	else:
		return data

inPath_original = '/home/nagfa5/GAN/04_groups/'
inPath_generated = '/home/nagfa5/GAN/04_analysis/'
outPath = '/home/nagfa5/GAN/05_stocha_a/'
regions = ['middle','north','south']

for i in range(3):
	print(regions[i])
	# load, rescale and compute mean of original
	original = rescale(np.load(inPath_original+'01_data_'+regions[i]+'.npy'))
	original = np.mean(original,axis=0)
	generated = rescale(np.load(inPath_generated+'generated_time_'+regions[i]+'.npy'))
	# relative error for time series
#	relative = np.empty(shape=generated.shape)
#	for j in range(relative.shape[0]):
#		for k in range(relative.shape[1]):
#			relative[j,k] = clip(abs(original[k]-generated[j,k])/original[k])
#	np.save(outPath+'relative_time_'+regions[i],relative)
	# spectrum for time series
	idx = int(sys.argv[1])
	original_ps= spectrum(original)
	gen_ps = spectrum(generated[idx,:])
	freq = np.fft.fftfreq(generated.shape[-1])
	np.save(outPath+'spectrum_'+regions[i],np.concatenate((original_ps,gen_ps),axis=0))
	np.save(outPath+'spectrum_freq_'+regions[i],freq)
