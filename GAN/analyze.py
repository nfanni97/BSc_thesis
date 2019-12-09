import numpy as np
import sys

def compute_center(data):
	center = np.empty(shape=(2))
	center[0] = np.mean(data[:,0])
	center[1] = np.mean(data[:,1])
	return center

def compute_diversions(data,center):
	diversions = np.empty(shape=(2))
	distance_mean = abs(data[:,0] - center[0])
	distance_var = abs(data[:,1] - center[1])
	diversions[0] = np.mean(distance_mean)
	diversions[1] = np.mean(distance_var)
	return diversions

def compute_correlation(data):
	corr = np.corrcoef(data[:,0],data[:,1])
	return np.array([corr[1,0]])

def compute_time_correlation(original_mean,generated):
	corrs = np.empty(shape=(generated.shape[0]))
	for i in range(generated.shape[0]):
		corrs[i] = np.corrcoef(original_mean,generated[i,:])[1,0]
	return np.array([np.mean(corrs)])

def create_vector(center,diversions,correlation,time_correlation):
	return np.concatenate((center,diversions,correlation,time_correlation))

category = sys.argv[1]
run_start = int(sys.argv[2])
run_stop = int(sys.argv[3])
loss_type = sys.argv[4]
base_path = '/home/nagfa5/GAN/05_stocha_d/'+loss_type+'_loss/'+category+'/'

# original
original = np.load(base_path+'original_mean_var.npy')
original_time = np.mean(np.load(base_path+'original_time.npy'),axis=0,keepdims=True)
o_center = compute_center(original)
o_diversions = compute_diversions(original,o_center)
o_corr = compute_correlation(original)
o_vector = create_vector(o_center,o_diversions,o_corr,np.asarray([1])) # correlation with itself is 1

distances = np.empty(shape=(run_stop-run_start+1))

for r in range(run_start,run_stop+1,1):
	path = base_path+'run_%d/'%r
	data = np.load(path+'mean_var.npy')
	data_time = np.load(path+'gen_time.npy')

	##########
	# CENTER #
	##########
	# average mean and var
	center = compute_center(data)

	##############
	# DIVERSIONS #
	##############
	# average diversion from mean and var centers
	diversions = compute_diversions(data,center)

	###############
	# CORRELATION #
	###############
	correlation = compute_correlation(data)

	####################
	# TIME CORRELATION #
	####################
	time_correlation = compute_time_correlation(original_time,data_time)

	# compute distance from original
	vector = create_vector(center,diversions,correlation,time_correlation)
	distances[r-1] = np.linalg.norm(vector-o_vector)

distances_to_save = np.array([np.amin(distances),np.mean(distances),np.amax(distances)])
np.save(base_path+'distances',distances_to_save)
print(distances)
