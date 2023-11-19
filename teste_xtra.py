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

fig, ax = plt.subplots(ncols=3, nrows=1,   dpi = 75)

cp0 = kseg.Kseg(1, 1, 1, 1000)
cp0.fitCurve(x)
ax[0].plot(x[:,0], x[:,1], 'o', label = ' base de dados')
cp0.plot_curve(ax[0])
ax[0].legend()

cp1 = kseg.Kseg(3, 1, 1, 1000)
cp1.fitCurve(x)
ax[1].plot(x[:,0], x[:,1], 'o', label = ' base de dados')
cp1.plot_curve(ax[1])
ax[1].legend()

cp2 = kseg.Kseg(5, 1, 1, 1000)
cp2.fitCurve(x)
ax[2].plot(x[:,0], x[:,1], 'o', label = ' base de dados')
cp2.plot_curve(ax[2])
ax[2].legend()