import pandas as pd
import os
import numpy as np
from numpy.random import randn
import matplotlib.pyplot as plt
import pylab as pl
from pykalman import KalmanFilter
rnd = np.random.RandomState(0)

base_folder = 'C:/Users/shires/DeepLabCutData/multi_whisker/'
file_name = 'deep-cut-resnet_50-95shuffle1-300000-for-task-multi-whisker139.h5'

data_frame = pd.read_hdf(os.path.join(base_folder, file_name))
scorer = np.unique(data_frame.columns.get_level_values(0))[0]
body_parts_to_plot = list(np.unique(data_frame.columns.get_level_values(1)))
n_frames = len(data_frame.index)

bp = 'whisker_2_4'

x = np.array(data_frame[scorer][bp]['x'])
y = np.array(data_frame[scorer][bp]['y'])
v = np.diff(np.insert(x, 0, x[0]))
a = np.diff(np.insert(v, 0, v[0]))

plt.plot(x)
plt.plot(a)
plt.show()

# generate a noisy sine wave to act as our fake observations
n_timesteps = len(x)
t = np.linspace(0, 1, len(x))
observations = np.vstack((x, y)).transpose()
print(observations.shape)

# create a Kalman Filter by hinting at the size of the state and observation
# space.  If you already have good guesses for the initial parameters, put them
# in here.  The Kalman Filter will try to learn the values of all variables.
# kf = KalmanFilter(transition_matrices=np.array([[1, 1], [0, 1]]),
#                   transition_covariance=0.01 * np.eye(2))

kf = KalmanFilter(n_dim_obs=2, n_dim_state=2)

# You can use the Kalman Filter immediately without fitting, but its estimates
# may not be as good as if you fit first.
states_pred = kf.em(observations).smooth(observations)[0]
print(states_pred)
print('fitted model: {0}'.format(kf))

# Plot lines for the observations without noise, the estimated position of the
# target before fitting, and the estimated position after fitting.
pl.figure(figsize=(16, 6))
obs_scatter = pl.scatter(t, observations[:, 0], marker='x', color='b',
                         label='observations')
position_line = pl.plot(t, states_pred[:, 0],
                        linestyle='-', marker='o', color='r',
                        label='position est.')
velocity_line = pl.plot(t, states_pred[:, 1],
                        linestyle='-', marker='o', color='g',
                        label='velocity est.')
pl.legend(loc='lower right')
pl.xlim(xmin=0, xmax=t.max())
pl.xlabel('time')
pl.show()