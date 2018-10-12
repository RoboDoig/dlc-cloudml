import pandas as pd
import os
import numpy as np
rnd = np.random.RandomState(0)
import scipy.signal as sig


# grab data
base_folder = 'C:/Users/shires/DeepLabCutData/multi_whisker/'

file_names = ['deep-cut-resnet_50-95shuffle1-300000-for-task-multi-whisker7.h5',
              'deep-cut-resnet_50-95shuffle1-300000-for-task-multi-whisker83.h5',
              'deep-cut-resnet_50-95shuffle1-300000-for-task-multi-whisker139.h5',
              'deep-cut-resnet_50-95shuffle1-300000-for-task-multi-whisker154.h5']

body_parts = ['whisker_2_1', 'whisker_2_2', 'whisker_2_3', 'whisker_2_4']
all_position = np.empty((1, len(body_parts*2)))
all_frames = 0
for file in file_names:
    data_frame = pd.read_hdf(os.path.join(base_folder, file))
    scorer = np.unique(data_frame.columns.get_level_values(0))[0]
    n_frames = len(data_frame.index)
    all_frames += n_frames

    # generate a position array for a whole whisker
    for bp, body_part in enumerate(body_parts):
        x = data_frame[scorer][body_part]['x']
        y = data_frame[scorer][body_part]['y']

        data_frame[scorer][body_part]['x'] = sig.medfilt(x, 21)
        data_frame[scorer][body_part]['y'] = sig.medfilt(y, 21)

    data_frame.to_hdf(os.path.join(base_folder, file.split('.')[0] + '-smoothed.h5'),
                      'df_with_missing', format='table', mode='w')
