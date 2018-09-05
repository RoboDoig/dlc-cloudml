import tensorflow as tf
import argparse
import os
import sys

from trainer import evaluate

FLAGS = None


def main(_):
    evaluate.evaluate(FLAGS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        help='Directory to find training data',
        default='../test_data/'
    )
    parser.add_argument(
        '--task',
        help='Task name',
        default='reaching'
    )
    parser.add_argument(
        '--date',
        help='Data date',
        default='Jan30'
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
        '--scorer',
        help='Who scored this dataset?',
        default='Mackenzie'
    )
    parser.add_argument(
        '--job_dir',
        help='Directory for training job output',
        default='../test_job/'
    )
    parser.add_argument(
        '--snapshot',
        help='Name of the model snapshot to evaluate',
        default='snapshot-300000'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)