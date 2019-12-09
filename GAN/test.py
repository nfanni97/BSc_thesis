import numpy as np

t1 = np.array([
    [1,2,3,4,5],
    [2,3,4,5,6]
])

t2 = np.array([
    [1,2,3,4,5],
    [2,3,4,5,6]
])

t3 = t1[:,2] + t2[:,2]

print(t3.shape)