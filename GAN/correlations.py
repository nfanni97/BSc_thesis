import numpy as np

inPath = '/home/nagfa5/GAN/04_groups/'
outPath = '/home/nagfa5/GAN/04_analysis/'
regions = ['middle','north','south']

def compute_mean(data):
	return np.mean(data,axis=0,keepdims=True)

def rescale(data):
	data_min = np.amin(data,axis=1,keepdims=True)
	data_max = np.amax(data,axis=1,keepdims=True)
	return (data - data_min) / (data_max - data_min)

# data_1 is the mean, data_2 contains observations: in the end, mean of those correlations has to be computed
def correlation_value(data_1,data_2):
	return np.corrcoef(data_1,data_2)[1,0]

corr_1 = np.empty(shape=(3))
corr_2 = np.empty(shape=(3))
corr_3 = np.empty(shape=(3,3))
corr_4 = np.empty(shape=(3))
original_means = np.empty(shape=(3,168))
generated = np.empty(shape=(3,100,168))
for i in range(3):
	print(regions[i])
	# 0. load, rescale and compute mean
	original = np.load(inPath+'01_data_'+regions[i]+'.npy')
	original = rescale(original)
	original_means[i,:] = compute_mean(original)

	temp_generated = np.load(outPath+'generated_time_'+regions[i]+'.npy')
	temp_generated = rescale(temp_generated)
	generated[i,:,:,] = temp_generated

	# 1. correlation between original and generated images (means)
	corrs = np.empty(shape=(generated.shape[1]))
	for j in range(generated.shape[1]):
		corrs[j] = correlation_value(original_means[i,:],generated[i,j,:])
	corr_1[i] = np.mean(corrs)

# 2. cross-correlation: between means of original series
for i in range(3):
	for j in range(i+1,3):
		print(regions[i]+', '+regions[j])
		corr_2[i] = correlation_value(original_means[i,:],original_means[j,:])
		print(corr_2[i])

# 3. cross correlation: between original and generated across regions
for i in range(3):
	for j in range(3):
		print('generated: %s, original: %s'%(regions[i],regions[j]))
		corrs = np.empty(shape=(generated.shape[1]))
		for k in range(generated.shape[1]):
			corrs[k] = correlation_value(original_means[j,:],generated[i,k,:])
		corr_3[i,j] = np.mean(corrs)

# 4. cross correlation: between generated series
for i in range(3):
	for j in range(i+1,3):
		print(regions[i]+', '+regions[j])
		corrs = np.empty(shape=(generated.shape[1],generated.shape[1]))
		for k in range(generated.shape[1]):
			for l in range(generated.shape[1]):
				corrs[i,j] = correlation_value(generated[i,k,:],generated[j,l,:])
		corr_4[i] = np.mean(corrs)
		print(corr_4[i])

np.save(outPath+'corr_1',corr_1)
np.save(outPath+'corr_2',corr_2)
np.save(outPath+'corr_3',corr_3)
np.save(outPath+'corr_4',corr_4)
