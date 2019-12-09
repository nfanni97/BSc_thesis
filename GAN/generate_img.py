import numpy as np
from keras.models import load_model
import sys

region = sys.argv[1]
category = sys.argv[2]
steps = sys.argv[3]
noise_dim = 100
num_img = 100
path = '/home/nagfa5/GAN/05_stocha_c/'+category+'/'

model = load_model(path+'models/gen_'+str(steps))
noise = np.random.uniform(low=0,high=1,size=[num_img,noise_dim])
generated = model.predict(noise)
np.save(path+'generated_img',generated)
