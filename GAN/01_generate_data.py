import numpy as np
from keras.models import load_model

#############
# FUNCTIONS #
#############

def generate_imgs(num,noise_dim,model_path,generated_path):
    generator = load_model(model_path)
    noise = np.random.uniform(low=0, high=1, size=[num, noise_dim])
    generated = generator.predict(noise)
    np.save(generated_path, generated)

# data is numpy array
def convert_to_time(data,GASF=True):
    time_series = np.zeros((data.shape[0],data.shape[1]))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            time_series[i,j] = np.cos(np.arccos(data[i,j,j]) / 2)
    return time_series

##############
# PARAMETERS #
##############

path = '/home/fanni/7.felev/Szakdoga/GAN/05_stocha_a/'
img_num = 100
noise_dim = 100
generated_name = 'north_generated_img'

#################
# GENERATE DATA #
#################

# generate_imgs(img_num,noise_dim,path+'north_gen_2000',generated_name)

################
# CONVERT DATA #
################

generated = np.load(path+generated_name+'.npy')
generated = convert_to_time(generated)

original = np.load('/home/fanni/Dokumentumok/PPKE-ITK/7.felev/Szakdoga/GAN/04_groups/01_data_north.npy')
#rescale
original_min = np.amin(original,axis=1,keepdims=True)
original_max = np.amax(original,axis=1,keepdims=True)
original =  (original - original_min) / (original_max - original_min)

# save data
np.save(path+'north_generated_time',generated)
np.save(path+'north_original_time',original)