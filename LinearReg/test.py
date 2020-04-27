import numpy as np
a=np.zeros((3, 4))
print(a)
b=np.mat(a)
print(b)
print(a.shape)
print(b.shape)
print(a[:,1].shape)
print(b[:,1].shape)