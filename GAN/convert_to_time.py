import numpy as np
import sys

category = sys.argv[1]
run_start = int(sys.argv[2])
run_stop = int(sys.argv[3])
loss_type = sys.argv[4]
file_name = 'gen_img'
base_path = '/home/nagfa5/GAN/05_stocha_d/'+loss_type+'_loss/'+category+'/'

# original data rescale:
original = np.load('/home/nagfa5/GAN/05_stocha_c/'+category+'/01_data_north_'+category+'.npy')
o_min = np.amin(original,axis=1,keepdims=True)
o_max = np.amax(original,axis=1,keepdims=True)
original = (original - o_min) / (o_max - o_min)
np.save(base_path+'original_time',original)

for r in range(run_start,run_stop+1,1):
	path = base_path + 'run_' + str(r) + '/'
	data = np.load(path+file_name+'.npy')
	converted = np.empty(shape=(data.shape[0],data.shape[1]))
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			converted[i,j] = np.cos(np.arccos(data[i,j,j])/2)
	# rescale
	min_val = np.amin(converted,axis=1,keepdims=True)
	max_val = np.amax(converted,axis=1,keepdims=True)
	converted = (converted - min_val) / (max_val - min_val)
	np.save(path+'gen_time',converted)
