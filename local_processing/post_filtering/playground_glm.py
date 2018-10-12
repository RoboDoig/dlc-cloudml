import pandas as pd
import os
import numpy as np
from numpy.random import randn
import pylab as pl
from pykalman import KalmanFilter
rnd = np.random.RandomState(0)
from pyglmnet import GLM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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

    position_array = np.empty((n_frames, len(body_parts)*2))

    # generate a position array for a whole whisker
    for bp, body_part in enumerate(body_parts):
        x = data_frame[scorer][body_part]['x']
        y = data_frame[scorer][body_part]['y']

        position_array[:, bp] = x
        position_array[:, bp+4] = y

    all_position = np.vstack((all_position, position_array))
all_position = all_position[1:, :]

# history dependence
history = 2
for i in reversed(range(history)):
    print(i)
    position_array = all_position[i:, :]

print(position_array.shape)
pl.figure()
for n in range(n_frames):
    pl.scatter(all_position[n, 0:4], all_position[n, 4:8], s=2, c='k')

pl.show()

# GLM
glm = GLM(distr='gaussian', alpha=0.05)

X = np.delete(all_position, 0, axis=1)
y = all_position[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler().fit(X_train)
glm.fit(scaler.transform(X_train), y_train)

yhat = glm.predict(scaler.transform(X))
# print(glm.score(X_test, Y_test))
#
# plot
pl.figure()
pl.plot(y, marker='x', color='b', label='observed')
pl.plot(yhat[9, :], marker='o', color='r', label='trained')

pl.show()
