import numpy as np
from keras.models import load_model

regions = ['middle','north','south']
inPath = '/home/nagfa5/GAN/04_groups/models/'
num_img = 100
noise_dim = 100

for i in range(3):
	print('working on '+regions[i])
	path = inPath+regions[i]+'/'
	model = load_model(path+'gen_2000')
	noise = np.random.uniform(low=0.0,high=1.0,size=(num_img,noise_dim))
	generated = model.predict(noise)[:,:,:,0]
	# convert to time
	converted = np.empty(shape=(generated.shape[0],generated.shape[1]))
	for j in range(generated.shape[0]):
		for k in range(generated.shape[1]):
			converted[j,k] = np.cos(np.arccos(generated[j,k,k])/2)
	np.save('/home/nagfa5/GAN/04_analysis/generated_time_'+regions[i],converted)
