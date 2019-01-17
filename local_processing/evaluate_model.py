import sys
import os
import numpy as np
import pandas as pd
from trainer.config import load_config
import trainer.predict as predict
from trainer.dataset.pose_dataset import data_to_input
from local_processing import auxiliary_functions
import pickle
from tqdm import tqdm
from skimage import io
import skimage.color

base_folder = 'Z:/Data/BarthAirPuff/'
task = 'air-puff'
date = 'Dec7'
shuffles = [1]
training_fraction = [0.95]
scorer = 'Andrew'


def evaluate_model():
    for shuffle_index, shuffle in enumerate(shuffles):
        for train_fraction_index, train_fraction in enumerate(training_fraction):
            experiment_name = task + date + '-trainset' + str(
                int(train_fraction * 100)) + 'shuffle' + str(shuffle)

            cfg = load_config(os.path.join(base_folder, experiment_name, 'test/pose_cfg.yaml'))

            # get available snapshots
            snapshots = np.array([
                fn.split('.')[0] for fn in os.listdir(
                    os.path.join(
                        base_folder, 'trained-results'
                    )
                ) if 'index' in fn
            ])

            # just pick most trained snapshot
            increasing_indices = np.argsort([int(m.split('-')[1]) for m in snapshots])
            snapshots = snapshots[increasing_indices]

            cfg['init_weights'] = os.path.join(base_folder, 'trained-results', snapshots[0])
            training_iterations = (cfg['init_weights'].split('/')[-1]).split('-')[-1]

            # load metadata for training / test files and labels
            dataset_folder = 'unaugmented-data-set-' + task + date
            doc_file = '/documentation_data-' + task + str(int(train_fraction * 100)) + \
                       'shuffle' + str(int(shuffle)) + '.pickle'
            with open(os.path.join(base_folder, dataset_folder + doc_file), 'rb') as f:
                data_doc, train_indices, test_indices, __ignore__ = pickle.load(f)

            data = pd.read_hdf(os.path.join(base_folder, dataset_folder, 'data-' + task,
                               'collected-data-' + scorer + '.h5'), 'df_with_missing')

            # Load and setup CNN part detector and configuration
            sess, inputs, outputs = predict.setup_pose_prediction(cfg)

            num_images = len(data.index)
            predict_data = np.zeros((num_images, 3 * len(cfg['all_joints_names'])))
            test_set = np.zeros(num_images)

            # compute predictions over images
            for image_index, image_name in tqdm(enumerate(data.index)):
                # get each image
                image = io.imread(os.path.join(base_folder, dataset_folder, image_name))
                image = skimage.color.gray2rgb(image)
                image_batch = data_to_input(image)

                # run into CNN
                outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
                scmap, locref = predict.extract_cnn_output(outputs_np, cfg)

                # extract maximum scoring location from the heatmap
                pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
                predict_data[image_index, :] = pose.flatten()

            dlc_scorer = 'deep-cut-' + str(cfg['net_type']) + '-' + \
                         str(int(train_fraction * 100)) + 'shuffle' + str(shuffle) + \
                         '-' + str(training_iterations) + '-for-task-' + task

            index = pd.MultiIndex.from_product(
                [[dlc_scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
                names=['scorer', 'bodyparts', 'coords']
            )

            # save results
            auxiliary_functions.attempt_to_make_folder(os.path.join(base_folder, 'trained-results/evaluation/'))
            data_machine = pd.DataFrame(
                predict_data,
                columns=index,
                index=data.index.values
            )
            data_machine.to_hdf(os.path.join(base_folder,
                                             'trained-results/evaluation',
                                             dlc_scorer + '.h5'),
                                'df_with_missing', format='table', mode='w')
            # data_machine.to_csv(os.path.join(base_folder,
            #                                  'trained-results/evaluation',
            #                                  dlc_scorer + '.csv'))

            print(data_machine)


if __name__ == '__main__':
    evaluate_model()




