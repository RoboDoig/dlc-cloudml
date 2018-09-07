import os
import numpy as np
import yaml
import pandas as pd
import shutil
from skimage import io
import pickle
import scipy.io as sio

from local_processing import auxiliary_functions

base_folder = 'C:/Users/shires/DeepLabCutData/cell_01_video/'
task = 'pole-whisking'
scorer = 'Andrew'
date = 'Sep6'
shuffles = [1]
training_fraction = [0.95]
body_parts = ['pole']


def split_trials(trial_index, train_fraction=0.8):
    train_size = int(len(trial_index) * train_fraction)
    shuffle = np.random.permutation(trial_index)
    test_indexes = shuffle[train_size:]
    train_indexes = shuffle[:train_size]

    return (train_indexes, test_indexes)


def box_it_into_a_cell(joints):
    outer = np.array([[None]], dtype=object)
    outer[0, 0] = np.array(joints, dtype='int64')
    return outer


def make_train_pose_yaml(items_to_change, save_as_file, filename='pose_cfg.yaml'):
    docs = []
    raw = open(filename).read()
    for raw_doc in raw.split('\n---'):
        try:
            docs.append(yaml.load(raw_doc))
        except SyntaxError:
            docs.append(raw_doc)

    for key in items_to_change.keys():
        docs[0][key] = items_to_change[key]

    with open(save_as_file, 'w') as f:
        yaml.dump(docs[0], f)

    return docs[0]


def make_test_pose_yaml(dictionary, keys_to_save, save_as_file):
    dict_test = {}
    for key in keys_to_save:
        dict_test[key] = dictionary[key]

    dict_test['scoremap_dir'] = 'test'
    with open(save_as_file, 'w') as f:
        yaml.dump(dict_test, f)


def generate_base_config(path):
    dict_base = {'dataset': '',
                 'num_joints': '',
                 'all_joints': '',
                 'all_joints_names': '',
                 'pos_dist_thresh': 17,
                 'global_scale': 0.8,
                 'scale_jitter_lo': 0.5,
                 'scale_jitter_up': 1.5,
                 'mirror': False,
                 'net_type': 'resnet_50',
                 'init_weights': '../../pretrained/resnet_v1_50.ckpt',
                 'location_refinement': True,
                 'locref_huber_loss': True,
                 'locref_loww_weight': 0.05,
                 'locref_stdev': 7.2801,
                 'intermediate_supervision': False,
                 'intermediate_supervision_layer': 12,
                 'max_input_size': 1000,
                 'multi_step': [[0.005, 1000], [0.02, 430000], [0.002, 730000], [0.001, 1030000]],
                 'display_iters': 1000,
                 'save_iters': 50000}

    with open(os.path.join(path, 'pose_cfg.yaml'), 'w') as f:
        yaml.dump(dict_base, f)


def generate_training_file():
    generate_base_config(base_folder)

    folder = os.path.join(base_folder, 'data-' + task)
    data = pd.read_hdf(folder + '/collected-data-' + scorer + '.h5',
                       'df_with_missing')[scorer]

    bf = os.path.join(base_folder, 'unaugmented-data-set-' + task + date + '/', 'data-' + task)
    shutil.copytree(folder, bf)

    for shuffle in shuffles:
        for train_fraction in training_fraction:
            train_indexes, test_indexes = split_trials(
                range(len(data.index)), train_fraction
            )
            filename_matfile = task + '_' + scorer + str(int(train_fraction * 100)) \
                               + 'shuffle' + str(shuffle)

            fn = os.path.join(base_folder, 'unaugmented-data-set-' + task + date +
                              '/documentation_' + 'data-' + task + str(int(train_fraction * 100))
                              + 'shuffle' + str(shuffle))

            # make matlab train file
            dat = []
            for jj in train_indexes:
                H = {}
                # load image to get dimensions
                filename = data.index[jj]
                im = io.imread(filename)
                H['image'] = filename

                if np.ndim(im) > 2:
                    H['size'] = np.array(
                        [np.shape(im)[2],
                         np.shape(im)[0],
                         np.shape(im)[1]]
                    )
                else:
                    H['size'] = np.array([1, np.shape(im)[0], np.shape(im)[1]])

                index_joints = 0
                joints = np.zeros((len(body_parts), 3)) * np.nan
                for bp_index, body_part in enumerate(body_parts):
                    if data[body_part]['x'][jj] < np.shape(im)[1] and data[body_part]['y'][jj] < np.shape(im)[0]:
                        joints[index_joints, 0] = int(bp_index)
                        joints[index_joints, 1] = data[body_part]['x'][jj]
                        joints[index_joints, 2] = data[body_part]['y'][jj]
                        index_joints += 1

                joints = joints[np.where(np.prod(np.isfinite(joints), 1))[0], :]

                assert (np.prod(np.array(joints[:, 2]) < np.shape(im)[0]))
                assert (np.prod(np.array(joints[:, 1]) < np.shape(im)[1]))

                H['joints'] = np.array(joints, dtype=int)
                if np.size(joints) > 0:
                    dat.append(H)

            with open(fn + '.pickle', 'wb') as f:
                pickle.dump([data, train_indexes, test_indexes, train_fraction], f,
                            pickle.HIGHEST_PROTOCOL)

            DTYPE = [('image', 'O'), ('size', 'O'), ('joints', 'O')]
            matlab_data = np.array(
                [(np.array([dat[item]['image']], dtype='U'),
                  np.array([dat[item]['size']]),
                  box_it_into_a_cell(dat[item]['joints']))
                  for item in range(len(dat))],
                dtype=DTYPE)

            sio.savemat(base_folder + '/unaugmented-data-set-' + task + date + '/'
                        + filename_matfile + '.mat', {'dataset': matlab_data})

            # Creating file structure for training and test files + pose_yaml
            experiment_name = os.path.join(base_folder, task + date + '-trainset' + str(
                                           int(train_fraction * 100)) + 'shuffle' + str(shuffle))

            auxiliary_functions.attempt_to_make_folder(experiment_name)
            auxiliary_functions.attempt_to_make_folder(experiment_name + '/train')
            auxiliary_functions.attempt_to_make_folder(experiment_name + '/test')

            items_to_change = {
                'dataset': '../../' + 'unaugmented-data-set-' + task + date + '/' + filename_matfile + '.mat',
                'num_joints': len(body_parts),
                'all_joints': [[i] for i in range(len(body_parts))],
                'all_joints_names': body_parts
            }

            training_data = make_train_pose_yaml(
                items_to_change,
                experiment_name + '/train/' + 'pose_cfg.yaml',
                filename=os.path.join(base_folder, 'pose_cfg.yaml')
            )
            keys_to_save = [
                'dataset', 'num_joints', 'all_joints', 'all_joints_names',
                'net_type', 'init_weights', 'global_scale', 'location_refinement',
                'locref_stdev'
            ]
            make_test_pose_yaml(training_data, keys_to_save,
                                experiment_name + '/test/' + 'pose_cfg.yaml')


generate_training_file()
