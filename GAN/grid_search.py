import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import sys
import warnings

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(data, arima_order):
	# prepare training dataset
	train_size = int(len(data) * 0.714)
	train, test = data[0:train_size], data[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		prediction = model_fit.forecast()[0]
		predictions.append(prediction)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values,category):
	dataset = dataset.astype('float32')
	best_score, best_cfg = float("inf"), None
	with open('/home/nagfa5/GAN/05_stocha_b/grid_search_result_'+category,'w+') as f:
		f.write('p,d,q,mse\n')
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					log_msg = 'ARIMA%s MSE=%.5f' % (order,mse)
					print(log_msg)
					with open('/home/nagfa5/GAN/05_stocha_b/grid_search_result_'+category,'a') as f:
						f.write('%d,%d,%d,%f\n' % (p,d,q,mse))
				except KeyboardInterrupt:
					sys.exit()
				except:
					continue
	print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))

# load data
category = sys.argv[1]
if category == 'loss':
	data = np.load('/home/nagfa5/GAN/05_stocha_a/north_original_time.npy')
else:
	data = np.load('/home/nagfa5/GAN/05_stocha_b/02_time_'+category+'_north.npy')

# compute mean so there's only one time series
data_min = np.amin(data,axis=0,keepdims=True)
data_max = np.amax(data,axis=0,keepdims=True)
data = (data - data_min) / (data_max - data_min)
data_mean = np.mean(data,axis=0)

# set values to be examined
p_values = [0,1,2,4,6,8,10,12,24]
d_values = range(0,3)
q_values = range(0,3)

# search
warnings.filterwarnings('ignore')
evaluate_models(data_mean,p_values,d_values,q_values,category)
