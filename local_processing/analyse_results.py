import os
import sys
import numpy as np
import pandas as pd
import local_processing.analysis_util.analysis_util as au
import pickle
from local_processing import auxiliary_functions
import matplotlib.pyplot as plt
from skimage import io

base_folder = 'D:/Andrew/AH08861015181x2/dataset-selection/'
task = 'multi-whisker-1x2'
date = 'Oct16'
shuffles = [1]
training_fraction = [0.95]
scorer = 'Andrew'
pcutoff = 0.3


def analyse_results():
    dataset_folder = 'unaugmented-data-set-' + task + date
    data = pd.read_hdf(os.path.join(base_folder, dataset_folder, 'data-' + task,
                       'collected-data-' + scorer + '.h5'), 'df_with_missing')

    comparison_body_parts = list(np.unique(data.columns.get_level_values(1)))
    colors = au.get_cmap(len(comparison_body_parts))

    for train_fraction in training_fraction:
        for shuffle in shuffles:
            fns = [file for file in os.listdir(os.path.join(base_folder, 'trained-results/evaluation'))
                   if 'for-task-' + task in file and 'shuffle' + str(shuffle) in file
                   and str(int(train_fraction * 100)) in file]

            doc_file = 'documentation_data-' + task + str(int(train_fraction * 100)) + \
                       'shuffle' + str(int(shuffle)) + '.pickle'
            doc_file = os.path.join(base_folder, dataset_folder, doc_file)
            with open(doc_file, 'rb') as f:
                [training_data_details,
                 train_indices,
                 test_indices,
                 test_fraction_data] = pickle.load(f)

            training_iterations = [(int(fns[j].split('-for-task-')[0].split('-')[-1]), j) for j in range(len(fns))]
            training_iterations.sort(key=lambda tup: tup[0])
            training_iterations = training_iterations

            for training_iteration, index in training_iterations:
                data_machine = pd.read_hdf(os.path.join(base_folder, 'trained-results/evaluation',
                                                        fns[index]),
                                           'df_with_missing')
                data_combined = pd.concat([data.T, data_machine.T], axis=0).T
                scorer_machine = data_machine.columns.get_level_values(0)[0]
                mse, mse_pcutoff = au.pairwise_distances(data_combined, scorer, scorer_machine,
                                                         pcutoff, comparison_body_parts)

                test_error = np.nanmean(mse.iloc[test_indices].values.flatten())
                train_error = np.nanmean(mse.iloc[train_indices].values.flatten())

                test_error_pcutoff = np.nanmean(mse_pcutoff.iloc[test_indices].values.flatten())
                train_error_pcutoff = np.nanmean(mse_pcutoff.iloc[train_indices].values.flatten())

                print('train error: ', np.round(train_error, 2), 'pixels --- test error: ', np.round(test_error, 2),
                      'pixels')
                print('train error with cutoff: ', np.round(train_error_pcutoff, 2),
                      'pixels --- test error with cutoff: ', np.round(test_error_pcutoff, 2), 'pixels')

                # plotting
                folder_name = os.path.join(base_folder, 'trained-results/evaluation/labeled')
                auxiliary_functions.attempt_to_make_folder(folder_name)
                num_frames = np.size(data_combined.index)
                for ind in np.arange(num_frames):
                    fn = data_combined.index[ind]

                    fig = plt.figure()
                    ax = fig.add_subplot(1, 1, 1)
                    au.make_labeled_image(data_combined,
                                          ind,
                                          os.path.join(base_folder, dataset_folder),
                                          [scorer, scorer_machine],
                                          comparison_body_parts,
                                          colors,
                                          pcutoff=pcutoff)

                    if ind in train_indices:
                        plt.savefig(os.path.join(folder_name, 'train-image' + str(ind) + '-'
                                                 + fn.split('/')[0] + '-' + fn.split('/')[1]))
                    else:
                        plt.savefig(os.path.join(folder_name, 'test-image' + str(ind) + '-'
                                                 + fn.split('/')[0] + '-' + fn.split('/')[1]))

                    plt.close('all')


if __name__ == '__main__':
    analyse_results()

