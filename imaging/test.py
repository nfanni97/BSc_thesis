import numpy as np

filename='source/train_data'

raw = open(filename).readlines()
raw = [list(map(float, each.strip().split())) for each in raw]
raw = np.array(raw)

print(raw.shape)