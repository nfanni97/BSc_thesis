import numpy as np
import matplotlib.pyplot as plt

def plot_mean_var(gen_mean,gen_var,original_mean,original_var):
    fig = plt.figure()
    plt.plot(gen_mean, gen_var, color='r', linestyle='', marker='o')
    plt.plot(original_mean, original_var, color='b', linestyle='', marker='s')
    plt.legend(['generated', 'original'])
    plt.xlabel('mean')
    plt.ylabel('var')
    plt.title('Mean and var of generated and original data')
    plt.show()

# data is a time series
def spectrum(data):
    ps = np.abs(np.fft.fft(data))**2
    return ps

path = '/home/fanni/7.felev/Szakdoga/GAN/05_stocha_a/'

gen_data = np.load(path+'north_generated_time.npy')
original_data = np.load(path+'north_original_time.npy')

############
# MEAN-VAR #
############

original_mean =  np.mean(original_data,axis=1)
original_var = np.var(original_data,axis=1)

gen_min = np.amin(gen_data,axis=1,keepdims=True)
gen_max = np.amax(gen_data,axis=1,keepdims=True)
gen_data = (gen_data - gen_min) / (gen_max - gen_min)

gen_mean = np.mean(gen_data,axis=1)
gen_var = np.var(gen_data,axis=1)

plot_mean_var(gen_mean,gen_var,original_mean,original_var)

fig = plt.figure()
plt.plot(original_data[0,:],color='r')
plt.plot(gen_data[0,:],color='b')
plt.legend(['original','generated'])
plt.title('One time series of generated and original data')
plt.show()

# ############
# # SPECTRUM #
# ############

original_index = np.random.randint(low=0,high=original_data.shape[0])
original_ps = spectrum(original_data[original_index,:])

gen_index = np.random.randint(low=0,high=gen_data.shape[0])
gen_ps = spectrum(gen_data[gen_index,:])

freq = np.fft.fftfreq(original_data.shape[-1])

# plot
plt.plot(freq,original_ps,freq,gen_ps)
plt.legend(['original','generated'])
plt.xlabel('Hz')
plt.ylabel('ampl^2')
plt.title('Power spectrum of generated and original data')
plt.show()

# save
np.save(path+'original_ps',original_ps)
np.save(path+'gen_ps',gen_ps)
np.save(path+'ps_freq',freq)

##################
# RELATIVE ERROR #
##################

# IMAGES

original_img_mean = np.load(path+'north_original_img_mean.npy')
original_img_mean = np.squeeze(original_img_mean)

gen_img = np.load(path+'north_generated_img.npy')
gen_img = np.squeeze(gen_img)

relative_errors_img = np.empty(shape=gen_img.shape) # relative error to mean of original data

for i in range(gen_img.shape[0]):
    for j in range(gen_img.shape[1]):
        for k in range(gen_img.shape[2]):
            # relative error for x_0 observation and x the true value: x_0 / x - 1
            relative_errors_img[i,j,k] =  (gen_img[i,j,k] / original_img_mean[j,k] - 1) * 100
            # squash outliers
            if relative_errors_img[i,j,k] > 100:
                relative_errors_img[i,j,k] = 100
            if relative_errors_img[i,j,k] < -100:
                relative_errors_img[i,j,k] = -100

# save
np.save(path+'north_relative_error_img',relative_errors_img)

# generated images are shifted: when the difference is plotted, the pattern remains
plt.figure()
plt.imshow(gen_img[0,:,:] - original_img_mean)
plt.colorbar()
plt.title('Difference between generated and original image')
plt.show()

# plot relative errors
plt.figure()
plt.imshow(relative_errors_img[0,:,:])
plt.colorbar()
plt.title('A relative error')
plt.show()

# TIME SERIES

original_mean_0 = np.mean(original_data,axis=0,keepdims=True)
relative_errors_time = np.empty(shape=gen_data.shape)

for i in range(gen_data.shape[0]):
    for j in range(gen_data.shape[1]):
        relative_errors_time[i,j] = (gen_data[i,j] / original_mean_0[0,j] - 1) * 100
        # squash outliers
        if relative_errors_time[i,j] > 100:
            relative_errors_time[i,j] = 100
        if relative_errors_time[i,j] < -100:
            relative_errors_time[i,j] = -100

# save
np.save(path+'relative_error_time',relative_errors_time)

# plot
idx = np.random.randint(low=0,high=gen_data.shape[0])
plt.figure()
plt.plot(gen_data[idx,:])
plt.plot(original_mean_0[0,:])
plt.plot(relative_errors_time[idx,:] / 100)
plt.legend(['generated','original mean','relative error'])
plt.title('Comparison of random time series with original mean')
plt.show()

#########
# ARIMA #
#########