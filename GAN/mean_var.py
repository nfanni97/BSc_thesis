import numpy as np
import sys

category = sys.argv[1]
run_start = int(sys.argv[2])
run_stop = int(sys.argv[3])
loss_type = sys.argv[4]
base_path = '/home/nagfa5/GAN/05_stocha_d/'+loss_type+'_loss/'+category+'/'

original = np.load(base_path+'original_time.npy')
# (mean, var) tuples
original_mean_var = np.empty(shape=(original.shape[0],2))
original_mean_var[:,0] = np.mean(original,axis=1)
original_mean_var[:,1] = np.var(original,axis=1)

np.save(base_path+'original_mean_var',original_mean_var)

for r in range(run_start,run_stop+1,1):
	path = base_path+'run_%d/'%r
	data = np.load(path+'gen_time.npy')
	mean_var = np.empty(shape=(data.shape[0],2))
	mean_var[:,0] = np.mean(data,axis=1)
	mean_var[:,1] = np.var(data,axis=1)
	np.save(path+'mean_var',mean_var)
