import tensorflow as tf
import argparse
import os
import sys

from local_processing.video_maker import analyse_video

FLAGS = None


def main(_):
    analyse_video.analyse_video(FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='Directory to find training data',
        default='C:/Users/shires/DeepLabCutData/multi_whisker_2/'
    )
    parser.add_argument(
        '--snapshot_dir',
        help='Directory to find training data',
        default='C:/Users/shires/DeepLabCutData/multi_whisker_2/'
    )
    parser.add_argument(
        '--task',
        help='Task name',
        default='multi-whisker'
    )
    parser.add_argument(
        '--date',
        help='Data date',
        default='Sep6'
    )
    parser.add_argument(
        '--train_fraction',
        help='Train fraction',
        default=0.95
    )
    parser.add_argument(
        '--shuffle',
        help='Shuffle',
        default=1
    )
    parser.add_argument(
        '--snapshot_index',
        help='Snapshot index',
        default=0
    )
    parser.add_argument(
        '--video_name',
        help='Directory to save logs',
        default='2.mp4'
    )
    parser.add_argument(
        '--frame_buffer',
        help='Frame buffer',
        default=10
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)