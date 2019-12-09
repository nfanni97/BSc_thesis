import numpy as np
import sys

def file_len(fname):
	i = 0
	with open(fname,'r') as f:
		for i,l in enumerate(f):
			pass
	return i + 1

raw_path = '/home/nagfa5/raw_files/'
path = '/home/nagfa5/GAN/05_stocha_b/'
region = sys.argv[1]

num_days =  7
period_start = np.arange(start=0,stop=363,step=7)
period_start *= 24
period_stop = period_start + num_days * 24
bldgs = [2,7] # LargeHotel and PrimarySchool, respectively

num_bldgs = len(bldgs)

lengths = {
	'middle': 114,
	'north': 166,
	'south': 372
}

needed_length = lengths[min(lengths)]
num_samples = needed_length * period_start.shape[0]

data = np.empty(shape=(num_bldgs, num_samples, num_days * 24), dtype=np.single)

for b in range(num_bldgs):
	file_name = region+'_'+str(bldgs[b])+'.csv'
	file_path = raw_path+file_name
	print('working on: '+file_name)
	# read everything from file
	with open(file_path) as f:
		length = file_len(file_path)
		file_content = [None] * length
		needed_content = np.empty(shape=(length * 52, num_days * 24))
		for count, line in enumerate(f):
			file_content[count] = line.split(',')
			for i in range(52):
				needed_content[count*52+i] = file_content[count][period_start[i]:period_stop[i]]
			needed_content = np.array(needed_content,dtype=np.single)
	# remove zero rows
	needed_content = needed_content[~np.all(needed_content<1e-6,axis=1)]
	# get num_samples sample out of each file
	for i in range(num_samples):
		data[b,i,:] = needed_content[i,:]

# write to file
np.save(path+'02_time_good_'+region,data[0,:,:])
np.save(path+'02_time_bad_'+region,data[1,:,:])

print('DONE')
