import numpy as np
import pandas as pd
import os
import sys

task = 'pole-whisking'
base_folder = 'C:/Users/shires/DeepLabCutData/cell_01_video/'
scorer = 'Andrew'
image_type = '.png'
body_parts = ['pole']
invisible_boundary = 10


def to_data_frame():
    data_combined = None
    files = [
        fn for fn in os.listdir(os.path.join(base_folder, 'data-' + task, 'selected/'))
        if ("img" in fn and image_type in fn and '_labelled not in fn')
    ]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    image_address = ['data-' + task + '/selected/' + f for f in files]
    # image_address = [base_folder + 'data-' + task + '/selected/' + f for f in files]
    data_one_folder = pd.DataFrame({'image name': image_address})

    frame, Frame = None, None
    for body_part in body_parts:
        data_file = os.path.join(base_folder, 'data-' + task, 'selected/', body_part)

        d_frame = pd.read_csv(data_file + '.csv', sep=None, engine='python')

        if d_frame.shape[0] != len(image_address):
            new_index = pd.Index(
                np.arange(len(files)) + 1, name='Slice'
            )
            d_frame = d_frame.set_index('Slice').reindex(new_index)
            d_frame = d_frame.reset_index()

        index = pd.MultiIndex.from_product(
            [[scorer], [body_part], ['x', 'y']],
            names=['scorer', 'bodyparts', 'coords']
        )

        x_rescaled = d_frame.X.values.astype(float)
        y_rescaled = d_frame.Y.values.astype(float)

        invisible_markers_mask = (x_rescaled < invisible_boundary) * (y_rescaled < invisible_boundary)
        x_rescaled[invisible_markers_mask] = np.nan
        y_rescaled[invisible_markers_mask] = np.nan

        if Frame is None:
            frame = pd.DataFrame(
                np.vstack([x_rescaled, y_rescaled]).T,
                columns=index,
                index=image_address,
            )
            Frame = frame
        else:
            frame = pd.DataFrame(
                np.vstack([x_rescaled, y_rescaled]).T,
                columns=index,
                index=image_address
            )
            Frame = pd.concat(
                [Frame, frame],
                axis=1
            )

    data_single_user = Frame

    # save frame
    output_dir = os.path.join(base_folder, 'data-' + task, 'collected-data-' + scorer)
    data_single_user.to_csv(output_dir + '.csv')

    data_single_user.to_hdf(
        output_dir + '.h5',
        'df_with_missing',
        format='table',
        mode='w'
    )


to_data_frame()




