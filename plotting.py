import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import genfromtxt

a = genfromtxt('sampling_frames_matrix.csv', delimiter=',')
plt.imshow(a, cmap='autumn', interpolation='nearest')
plt.colorbar()
plt.title('heatmap:  n_samplig(2000) - n_frames(1000)')
plt.show()
