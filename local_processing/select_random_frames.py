import imageio
import requests
imageio.plugins.ffmpeg.download()
import os
from moviepy.editor import VideoFileClip
import numpy as np
from skimage.util import img_as_ubyte
from skimage import io

from local_processing import auxiliary_functions, frame_selection_tools

task = 'multi-whisker'
video_path = 'C:/Users/shires/DeepLabCutData/multi_whisker/'
video_type = '.mp4'
filename = 'all'
x1 = 0
x2 = 0
y1 = 0
y2 = 0
start = 0
stop = 1.0
date = 'Sep6'
cropping = False
num_frames_to_pick = 3
selection_algorithm = 'uniform'
check_cropping = False


def select_frames():
    if start > 1.0 or stop > 1.0 or start < 0 or stop < 0 or start >= stop:
        raise ValueError('Please change start & stop, they should form a '
                         'normalized interval with 0 <= start < stop <= 1.')
    else:
        base_folder = os.path.join(video_path, 'data-' + task + '/')
        auxiliary_functions.attempt_to_make_folder(base_folder)
        videos = auxiliary_functions.get_video_list(filename, video_path, video_type)
        for vindex, video in enumerate(videos):
            print("Loading ", video, '# ', str(vindex), ' of ', str(len(videos)))
            clip = VideoFileClip(os.path.join(video_path, video))
            # print("Duration of video [s], ", clip.duration, "fps, ", clip.fps,
            #       "Cropped frame dimensions: ", clip.size)

            # Create folder with experiment name and extract random frames
            folder = 'selected'
            v_name = video.split('.')[0]
            auxiliary_functions.attempt_to_make_folder(os.path.join(base_folder, folder))
            index_length = int(np.ceil(np.log10(clip.duration * clip.fps)))

            # extract first frame (uncropped) - useful for data augmentation
            # index = 0
            # image = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))
            # io.imsave(os.path.join(base_folder, folder, 'img' + v_name + '-'
            #                        + str(index).zfill(index_length) + '.png'), image)

            if cropping is True:
                clip = clip.crop(y1=y1, y2=y2, x1=x1, x2=x2)

            # print("Extracting frames")
            if selection_algorithm == 'uniform':
                frames_to_pick = frame_selection_tools.uniform_frames(clip, num_frames_to_pick, start, stop)
            elif selection_algorithm == 'kmeans':
                frames_to_pick = frame_selection_tools.k_means_based_frame_selection(clip, num_frames_to_pick,
                                                                                     start, stop)
            else:
                print('not implemented')
                frames_to_pick = []

            index_length = int(np.ceil(np.log10(clip.duration * clip.fps)))
            for index in frames_to_pick:
                try:
                    image = img_as_ubyte(clip.get_frame(index * 1. / clip.fps))
                    io.imsave(os.path.join(base_folder, folder, 'img' + v_name + '-'
                                           + str(index).zfill(index_length) + '.png'), image)
                except FileExistsError:
                    print('Frame # ', index, ' does not exist.')

            clip.close()


if __name__ == '__main__':
    select_frames()

