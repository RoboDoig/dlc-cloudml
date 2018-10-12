import pandas as pd
import os
import numpy as np
rnd = np.random.RandomState(0)
import scipy.signal as sig
import pylab as pl


def get_part_positions(file, constitutent_markers):
    data_frame = pd.read_hdf(file)
    scorer = np.unique(data_frame.columns.get_level_values(0))[0]
    n_frames = len(data_frame.index)

    position_array = np.empty((n_frames, len(constitutent_markers)*2))

    for bp, body_part in enumerate(constitutent_markers):
        x = data_frame[scorer][body_part]['x']
        y = data_frame[scorer][body_part]['y']

        position_array[:, bp] = x
        position_array[:, bp+len(constitutent_markers)] = y

    return position_array


# grab data
base_folder = 'C:/Users/shires/DeepLabCutData/multi_whisker/'
file_name = 'deep-cut-resnet_50-95shuffle1-300000-for-task-multi-whisker139.h5'
body_parts = ['whisker_2_1', 'whisker_2_2', 'whisker_2_3', 'whisker_2_4']

position_array = get_part_positions(os.path.join(base_folder, file_name), body_parts)
position_pole = get_part_positions(os.path.join(base_folder, file_name), ['pole1'])
n_frames = position_array.shape[0]

pl.figure()
for n in range(0, n_frames, 20):
    pl.scatter(position_array[n, 0:4], position_array[n, 4:8], s=2, c='k')
    pl.scatter(position_pole[n, 0], position_pole[n, 1], s=20, c='b')

    p_fit = np.polyfit(position_array[n, 0:4], position_array[n, 4:8], 2)

    x = np.linspace(position_array[n, 3]-1, position_array[n, 0], 100)
    y = (x**2 * p_fit[0]) + (x * p_fit[1]) + p_fit[0]

    y = y - (y[-1] - position_array[n, 4])

    pl.plot(x, y, c='k')

pl.show()