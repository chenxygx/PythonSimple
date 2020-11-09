import numpy as np

a = np.arange(15).reshape(3, 5)
np.savetxt('a.txt', a)
b = np.loadtxt('a.txt')
print(b)
