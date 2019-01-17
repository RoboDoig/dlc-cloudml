import pandas as pd
import os

base_folder = 'Z:/Data/BarthAirPuff/'
data_file = 'deep-cut-resnet_50-95shuffle1-600000-for-task-air-puff9 psi'

data_frame = pd.read_hdf(os.path.join(base_folder, data_file + '.h5'))
data_frame.to_csv(os.path.join(base_folder, data_file + '.csv'))


