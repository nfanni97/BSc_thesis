import numpy as np
from scipy import stats
import statistics

#############
# FUNCTIONS #
#############

def load_original_data(path,regions):
    result = {}
    for region in regions:
        result[region] = np.load(path+'01_data_'+region+'.npy')
        result[region] = rescale(result[region])
    return result

#to [0,1]
def rescale(data):
    data_min = np.amin(data,axis=1,keepdims=True)
    data_max = np.amax(data,axis=1,keepdims=True)
    data_normed = (data - data_min) / (data_max - data_min)
    return data_normed

#(168,168) -> (168), GASF
def img_to_times_series(img):
    time_series = np.empty((img.shape[0]))
    for i in range(img.shape[0]):
        time_series[i] = np.arccos(img[i,i]) / 2
    return time_series

#load data and transform back to time series
def load_generated_data(path,regions):
    #load data into a map with regions as keys and generated images (4x168x168 matrices) as values
    img = {}
    for region in regions:
        img[region] = np.load(path+'generated_2000_'+region+'.npy')
    #transform to time series
    time_series = {}
    for region in regions:
        time_series[region] = np.empty((img[region].shape[0],img[region].shape[1]))
        for i in range(img[region].shape[0]):
            time_series[region][i] = img_to_times_series(img[region][i])
    return time_series

def cross_correlation(data,regions,isOriginal):
    result = {}
    if isOriginal:
        for i in range(len(regions)):
            for j in range(i+1,len(regions)):
                r1 = regions[i][0]
                r2 = regions[j][0]
                if isOriginal:
                    result[r1+r2] = np.corrcoef(data[regions[i]],data[regions[j]])
    else:
        mean = compute_numpy(data,'mean',0,True)
        result = cross_correlation(mean,regions,True)
    return result

def save_data(path,data,name):
    for regions,data_slice in data.items():
        np.save(path+regions+'_'+name,data_slice)

def compute_numpy(data,func,axis,keepdims):
    if func == 'mean':
        return {
            region: np.mean(data_slice,axis=axis,keepdims=keepdims)
            for (region, data_slice) in data.items()
        }
    elif func == 'var':
        return {
            region: np.var(data_slice,axis=axis,keepdims=keepdims)
            for (region,data_slice) in data.items()
        }

def compute_scipy(data,func,axis):
    if func == 'mode':
        return {
            region: stats.mode(data_slice,axis=axis)[0]
            for region, data_slice in data.items()
        }

def compute_median(data):
    result = {}
    for region, data_slice in data.items():
        s = data_slice.shape[1]
        if s % 2 == 0:
            result[region] = (data_slice[:,s//2-1] + data_slice[:,s//2]) / 2
        else:
            result[region] = data_slice[:,s//2-1]
    return result

##############
# PARAMETERS #
##############

inPath = '/home/fanni/Dokumentumok/PPKE-ITK/7.felev/Szakdoga/GAN/04_groups/'
outPath = '/home/fanni/Dokumentumok/PPKE-ITK/7.felev/Szakdoga/GAN/05_stocha/'
regions = ['middle','north','south']

#####################
# CROSS-CORRELATION #
#####################

#maps with region names as keys
#load and rescale to [0,1]
original_data = load_original_data(inPath,regions)
#load and transform to time series
generated_data = load_generated_data(inPath,regions)

#compute mean of original
original_mean_0= compute_numpy(original_data,'mean',0,True)

#compute cross correlations
original_cross = cross_correlation(original_mean_0,regions,True)
generated_cross = cross_correlation(generated_data,regions,False)

save_data(outPath,original_cross,'original_cross_correlation')
save_data(outPath,generated_cross,'generated_cross_correlation')

################
# BASIC-STOCHA #
################

#compute mean and variance for original data on axis 1 (for every hour)
original_mean_1 = compute_numpy(original_data,'mean',1,True)
original_var_1 = compute_numpy(original_data,'var',1,True)

#compute mean and variance for generated data
generated_mean_1 = compute_numpy(generated_data,'mean',1,True)
generated_var_1 = compute_numpy(generated_data,'var',1,True)

#compute mode and median
generated_mode_1 = compute_scipy(generated_data,'mode',1)
original_mode_1 = compute_scipy(original_data,'mode',1)
generated_median_1 = compute_median(generated_data)
original_median_1 = compute_median(original_data)

#save data
save_data(outPath,original_mean_1,'original_mean_1')
save_data(outPath,original_var_1,'original_var_1')
save_data(outPath,generated_mean_1,'generated_mean_1')
save_data(outPath,generated_var_1,'generated_var_1')

save_data(outPath,generated_mode_1,'generated_mode_1')
save_data(outPath,original_mode_1,'original_mode_1')
save_data(outPath,generated_median_1,'generated_median_1')
save_data(outPath,original_median_1,'original_median_1')

############
# SPECTRUM #
############

