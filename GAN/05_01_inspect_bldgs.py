import numpy as np
import ast

path = '/home/nagfa5/raw_files/'
region = 'north'
num_days = 7
# get samples from january 1st to january 7th
period_start = 0
period_stop = period_start + num_days * 24
# read building names
with open('/home/nagfa5/Bldg_types') as f:
	b = f.readline()
bldg_names = ast.literal_eval(b)

for b in range(16):
	fileName = region+'_'+str(b)+'.csv'
	filePath = path+fileName
	print('Working on: '+fileName)
	# read first row and transform it to numpy array
	with open(filePath) as f:
		line = f.readline()
	data = line.split(',')
	data = np.array(data[period_start:period_stop],dtype=np.single)
	np.save('/home/nagfa5/GAN/05_stocha_b/north_'+str(b),data)
