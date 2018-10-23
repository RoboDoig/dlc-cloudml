import tensorflow as tf
import argparse
import sys

from local_processing import analyse_video

FLAGS = None


def main(_):
    analyse_video.analyse_video(FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='Directory to find training data',
        default='D:/Andrew/AH08861015181x2/dataset-selection/'
    )
    parser.add_argument(
        '--snapshot_dir',
        help='Directory to find training data',
        default='D:/Andrew/AH08861015181x2/dataset-selection/trained-results/'
    )
    parser.add_argument(
        '--task',
        help='Task name',
        default='multi-whisker-1x2'
    )
    parser.add_argument(
        '--date',
        help='Data date',
        default='Oct16'
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
        help='Video to process',
        default='288.mp4'
    )
    parser.add_argument(
        '--frame_buffer',
        help='Frame buffer',
        default=10
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)