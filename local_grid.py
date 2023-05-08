import pickle
import numpy as np
import cv2
file = open('local_grid.obj', 'rb')
data = pickle.load(file)
file.close()


rle_counts = []
new_data = []
for i in data:
    new_data[i] = data[i]*rle_counts[i]

b = np.frombuffer(new_data, dtype=np.int16)

print(rle_counts)
#grid = b.reshape(128, 128)
#print(grid)