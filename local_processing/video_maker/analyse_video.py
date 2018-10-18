import os
import numpy as np
import pandas as pd
import time
import pickle

from trainer.config import load_config
import trainer.predict as predict
from moviepy.editor import VideoFileClip
from skimage.util import img_as_ubyte
from tqdm import tqdm
import local_processing.analysis_util.analysis_util as au

# base_folder = 'C:/Users/shires/DeepLabCutData/multi_whisker_2/'
# task = 'multi-whisker'
# date = 'Sep6'
# shuffle = 1
# train_fraction = 0.95
# snapshot_index = 0
# video_name = '139.mp4'
# frame_buffer = 10


def analyse_video(FLAGS):
    experiment_name = FLAGS.task + FLAGS.date + '-trainset' + str(
        int(FLAGS.train_fraction * 100)) + 'shuffle' + str(FLAGS.shuffle)
    cfg = load_config(os.path.join(FLAGS.data_dir, experiment_name, 'test/pose_cfg.yaml'))

    # get available snapshots
    snapshots = np.array([
        fn.split('.')[0] for fn in os.listdir(
            os.path.join(
                FLAGS.snapshot_dir, 'trained-results'
            )
        ) if 'index' in fn
    ])
    increasing_indices = np.argsort([int(m.split('-')[1]) for m in snapshots])
    snapshots = snapshots[increasing_indices]

    # setup prediction over images
    cfg['init_weights'] = os.path.join(FLAGS.snapshot_dir, 'trained-results/' + snapshots[FLAGS.snapshot_index])

    training_iterations = (cfg['init_weights'].split('/')[-1].split('-')[-1])

    scorer = 'deep-cut-' + str(cfg['net_type']) + '-' + \
              str(int(FLAGS.train_fraction * 100)) + 'shuffle' + str(FLAGS.shuffle) + \
              '-' + str(training_iterations) + '-for-task-' + FLAGS.task

    sess, inputs, outputs = predict.setup_pose_prediction(cfg)
    pd_index = pd.MultiIndex.from_product(
        [[scorer], cfg['all_joints_names'], ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords']
    )

    # data definition
    video = os.path.join(FLAGS.data_dir, FLAGS.video_name)

    # do analysis
    clip = VideoFileClip(video)
    ny, nx = clip.size
    fps = clip.fps
    n_frames_approx = int(np.ceil(clip.duration * clip.fps) + FLAGS.frame_buffer)
    n_frames = n_frames_approx

    start = time.time()
    predict_data = np.zeros((n_frames_approx, 3 * len(cfg['all_joints_names'])))
    clip.reader.initialize()

    for index in tqdm(range(n_frames_approx)):
        image = img_as_ubyte(clip.reader.read_frame())

        if index == int(n_frames_approx - FLAGS.frame_buffer * 2):
            last_image = image
        elif index > int(n_frames_approx - FLAGS.frame_buffer * 2):
            if (image == last_image).all():
                n_frames = index
                break
            else:
                last_image = image

        pose = au.get_pose(image, cfg, inputs, outputs, sess)
        predict_data[index, :] = pose.flatten()

    stop = time.time()

    dictionary = {
        'start': start,
        'stop': stop,
        'run_duration': stop - start,
        'scorer': scorer,
        'config_file': cfg,
        'fps': fps,
        'frame_dimensions': (ny, nx),
        'nframes': n_frames
    }
    metadata = {'data': dictionary}

    data_machine = pd.DataFrame(predict_data[:n_frames, :], columns=pd_index, index=range(n_frames))
    data_machine.to_hdf(os.path.join(FLAGS.data_dir, scorer + FLAGS.video_name.split('.')[0] + '.h5'),
                        'df_with_missing', format='table', mode='w')

    with open(os.path.join(FLAGS.base_folder, scorer + FLAGS.video_name.split('.')[0] + '-metadata' + '.pickle'), 'wb') as f:
        pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)

#
# if __name__ == '__main__':
#     analyse_video()

