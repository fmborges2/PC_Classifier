import Kseg_new as kseg
import numpy as np
import matplotlib.pyplot as plt

dataset_name = 'Flame Norm'
import scipy.io as sio 
df = sio.loadmat('flame_data.mat')
df_x = df['X']
df_y = df['y']
X = np.concatenate((df_x, df_y), axis = 1)

x = X[0:1000, 0:2]
cp = kseg.Kseg(10, 1, 1, 1000)

cp.fitCurve(x)

fig, ax = plt.subplots(dpi = 150)
plt.plot(x[:,0], x[:,1], 'o')
cp.plot_curve(ax)
ax.legend()

