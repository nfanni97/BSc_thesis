import numpy as np

inPath = '/home/nagfa5/GAN/04_groups/'
outPath = '/home/nagfa5/GAN/04_analysis/'

def save_data(data,name):
	np.save(outPath+name+'.npy',data)

#load data
def load_data(region):
	return np.load(inPath+'02_data_'+region+'.npy')

middle_data = load_data('middle')
north_data = load_data('north')
south_data = load_data('south')

#compute mean and variance for all regions
middle_mean = np.mean(middle_data,axis=0)
north_mean = np.mean(north_data,axis=0)
south_mean = np.mean(south_data,axis=0)

save_data(middle_mean,'middle_mean')
save_data(north_mean,'north_mean')
save_data(south_mean,'south_mean')

middle_var = np.var(middle_data,axis=0)
north_var = np.var(north_data,axis=0)
south_var = np.var(south_data,axis=0)

#load generated data
def load_gen_data(region,step):
	return np.load(inPath+'generated/'+region+'/'+str(step)+'.npy')

middle_data_gen = load_gen_data('middle',2000)
north_data_gen = load_gen_data('north',2000)
south_data_gen = load_gen_data('south',2000)

#compute MSE
def compute_MSE(mean_data,gen_data):
	mse = np.zeros((gen_data.shape[1],gen_data.shape[2]))
	for i in range(gen_data.shape[0]):
		new_addition = np.subtract(mean_data,gen_data[i,:])
		new_addition = np.square(new_addition)
		mse = np.add(mse,new_addition)
	mse = np.divide(mse,gen_data.shape[0])
	return mse

middle_mse = compute_MSE(middle_mean,middle_data_gen)
north_mse = compute_MSE(north_mean,north_data_gen)
south_mse = compute_MSE(south_mean,south_data_gen)

#save data
save_data(middle_mse,'middle_mse')
save_data(north_mse,'north_mse')
save_data(south_mse,'south_mse')
