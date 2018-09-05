import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage import io
import skimage.color

from trainer.config import load_config
from trainer import predict
from trainer.dataset.pose_dataset import data_to_input
from tensorflow.python.lib.io import file_io


def evaluate(flags):
    cfg_path = os.path.join(flags.data_dir, flags.task + flags.date + '-trainset' +
                            str(int(flags.train_fraction * 100)) +
                            'shuffle' + str(int(flags.shuffle)) +
                            '/test/pose_cfg.yaml')

    # adjust config file and get directories
    cfg = load_config(cfg_path)
    cfg['dataset'] = cfg['dataset'].replace('../','')
    cfg['dataset'] = os.path.join(flags.data_dir, cfg['dataset'])

    data_folder = os.path.dirname(cfg['dataset'])
    data_file = 'Documentation_data-' + flags.task + '_' + str(int(flags.train_fraction * 100)) \
               + 'shuffle' + str(int(flags.shuffle)) + '.pickle'

    # load meta data / i.e. training & test file & labels
    with file_io.FileIO(os.path.join(data_folder, data_file), 'rb') as f:
        data, train_indices, test_indices, __ignore__ = pickle.load(f)

    with file_io.FileIO(os.path.join(data_folder, 'data-' + flags.task,
                          'CollectedData_' + flags.scorer + '.h5'), 'rb') as f:
        pd_data = pd.read_hdf(f)

    # load and setup CNN part detector #
    cfg['init_weights'] = os.path.join(flags.job_dir, flags.snapshot)
    training_iterations = (cfg['init_weights'].split('/')[-1]).split('-')[-1]
    dlc_scorer = 'DeepCut' + '_' + str(cfg['net_type']) + '_' + \
                 str(int(flags.train_fraction * 100)) + 'shuffle' + str(int(flags.shuffle)) + \
                 '_' + str(training_iterations) + 'forTask_' + flags.task

    # specify state of the model (snapshot / training state)
    sess, inputs, outputs = predict.setup_pose_prediction(cfg)

    num_images = len(pd_data.index)
    predict_data = np.zeros((num_images, 3 * len(cfg['all_joints_names'])))
    test_set = np.zeros(num_images)

    # compute predictions over images
    for image_index, image_name in tqdm(enumerate(pd_data.index)):
        image = io.imread(os.path.join(data_folder, 'data-' + flags.task, image_name), mode='RGB')
        image = skimage.color.gray2rgb(image)
        image_batch = data_to_input(image)

        # compute prediction with CNN
        outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
        scmap, locref = predict.extract_cnn_output(outputs_np, cfg)

        # extract maximum scoring location from heatmap
        pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
        predict_data[image_index, :] = pose.flatten()

    index = pd.MultiIndex.from_product([[dlc_scorer],
                                        cfg['all_joints_names'],
                                        ['x', 'y', 'likelihood']],
                                       names=['scorer', 'bodyparts', 'coords'])

    # save results
    data_machine = pd.DataFrame(predict_data, columns=index, index=pd_data.index.values)
    data_machine.to_hdf(os.path.join(flags.job_dir, dlc_scorer + '.h5'),
                        'df_with_missing', format='table', mode='w')
