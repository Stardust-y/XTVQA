from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np

x = np.array([1, 3, 4, 4])
y = np.array([4, 4, 4])

distance, path = fastdtw(x, y)

print(distance)
print(path)

# 5.0
# [(0, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 5), (3, 6), (4, 7)]
