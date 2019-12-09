import numpy as np
from keras.models import load_model
import sys

category = sys.argv[1]
run_start = int(sys.argv[2])
run_stop = int(sys.argv[3])
loss_type = sys.argv[4]

base_path = '/home/nagfa5/GAN/05_stocha_d/' + loss_type + '_loss/' + category + '/'
noise_dim = 100
num_of_img = 100

for r in range(run_start,run_stop+1,1):
	print('working on run %d of category %s' % (r,category))
	path = base_path + 'run_' + str(r) + '/'
	model = load_model(path+'gen_2000')
	noise = np.random.uniform(low=0,high=1,size=(noise_dim,num_of_img))
	generated = model.predict(noise)
	np.save(path+'gen_img',generated)
