import os
import pandas as pd
from skimage import io
import numpy as np

import matplotlib.pyplot as plt
from local_processing import auxiliary_functions

colormap = 'cool'
body_parts = ['pole', 'whiskerbase']
task = 'pole-whisking'
base_folder = 'C:/Users/shires/DeepLabCutData/cell_01_video/'
scorer = 'Andrew'
image_type = '.png'
scale = 1
Labels = ['.', '+', '*']
alpha_value = .6
m_size = 10


def get_cmap(n, name=colormap):
    return plt.cm.get_cmap(name, n)


def check_labels():
    color_scheme = get_cmap(len(body_parts))
    num_joints = len(body_parts)
    all_joints = map(lambda j: [j], range(num_joints))
    all_joints_names = body_parts
    num_body_parts = len(body_parts)

    data_combined = None

    data_combined = pd.read_hdf(os.path.join(base_folder, 'data-' + task,
                                             'collected-data-' + scorer + '.h5'),
                                'df_with_missing')

    folder = os.path.join(base_folder, 'data-' + task + '/', 'selected')
    print(folder)
    tmp_folder = folder + '-labeled'
    auxiliary_functions.attempt_to_make_folder(tmp_folder)

    files = [
        fn for fn in os.listdir(folder)
        if (image_type in fn and '_labeled' not in fn)
    ]
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    comparison_body_parts = body_parts

    for index, image_name in enumerate(files):
        image = io.imread(os.path.join(folder, image_name))
        plt.axis('off')

        if np.ndim(image) == 2:
            h, w = np.shape(image)
        else:
            h, w, cn = np.shape(image)

        plt.figure(frameon=False, figsize=(w * 1. / 100 * scale, h * 1. / 100 * scale))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        im_index = np.where(np.array(data_combined.index.values) == 'data-' + task + '/selected/' + image_name)[0]

        plt.imshow(image, 'bone')

        if index == 0:
            print('Creating images with labels by', scorer)
        for c, bp in enumerate(comparison_body_parts):
            plt.plot(
                data_combined[scorer][bp]['x'].values[im_index],
                data_combined[scorer][bp]['y'].values[im_index],
                Labels[0],
                color=color_scheme(c),
                alpha=alpha_value,
                ms=m_size
            )

        plt.xlim(0, w)
        plt.ylim(0, h)
        plt.axis('off')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.gca().invert_yaxis()
        plt.savefig(tmp_folder + '/' + image_name)

        plt.close('all')


check_labels()
