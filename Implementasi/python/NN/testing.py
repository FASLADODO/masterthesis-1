import random
import numpy as np

sizes = [5, 3, 2]



biases = [np.random.randn(y,1) for y in sizes[1:]]

print(biases)
weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

print(weights)

nabla_b = [np.zeros(b.shape) for b in biases]
nabla_w = [np.zeros(w.shape) for w in weights]

print(np.zeros((3,5)))
