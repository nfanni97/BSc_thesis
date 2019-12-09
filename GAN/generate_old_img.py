import numpy as np
from keras.models import load_model
import sys

region = sys.argv[1]
category = sys.argv[2]
steps = sys.argv[3]
noise_dim = 100
num_img = 100
cat_map = {'hotel':'good','school':'bad'}
path = '/home/nagfa5/GAN/05_stocha_b/'
out_path = '/home/nagfa5/GAN/05_stocha_c/'+category+'/'

model = load_model(path+'models/'+cat_map[category]+'/gen_'+str(steps))
noise = np.random.uniform(low=0,high=1,size=[num_img,noise_dim])
generated = model.predict(noise)
np.save(out_path+'old_generated_img',generated)
