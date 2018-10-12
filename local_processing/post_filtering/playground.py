import pandas as pd
import os
import numpy as np
from numpy.random import randn
import pylab as pl
from pykalman import KalmanFilter
rnd = np.random.RandomState(0)

base_folder = 'C:/Users/shires/DeepLabCutData/multi_whisker/'
file_name = 'deep-cut-resnet_50-95shuffle1-300000-for-task-multi-whisker139.h5'

data_frame = pd.read_hdf(os.path.join(base_folder, file_name))
scorer = np.unique(data_frame.columns.get_level_values(0))[0]
n_frames = len(data_frame.index)

body_parts = ['whisker_2_1', 'whisker_2_2', 'whisker_2_3', 'whisker_2_4']
position_array = np.empty((n_frames, len(body_parts)*2))

# generate a position array for a whole whisker
for bp, body_part in enumerate(body_parts):
    x = data_frame[scorer][body_part]['x']
    y = data_frame[scorer][body_part]['y']

    position_array[:, bp] = x
    position_array[:, bp+4] = y

print(position_array.shape)
pl.figure()
for n in range(n_frames):
    pl.plot(position_array[n, 0:4], position_array[n, 4:8], c='k')

# setup a Kalman filter
n_timesteps = position_array.shape[0]
t = np.linspace(0, 1, n_timesteps)

observations = position_array

kf = KalmanFilter(n_dim_obs=8, n_dim_state=8)
states_pred = kf.em(observations).filter(observations)[0]

pl.figure(figsize=(16, 6))
obs_scatter = pl.scatter(t, observations[:, 3], marker='x', color='b',
                         label='observations')
position_line = pl.plot(t, states_pred[:, 3],
                        linestyle='-', marker='o', color='r',
                        label='position est.')
pl.legend(loc='lower right')
pl.xlim(xmin=0, xmax=t.max())
pl.xlabel('time')
pl.show()
