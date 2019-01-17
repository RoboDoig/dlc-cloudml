import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import cv2

from skimage.draw import circle, line, bezier_curve
import skvideo
skvideo.setFFmpegPath('C:/Program Files/ffmpeg/bin/')
from local_processing.video_maker.video_processor import VideoProcessorSK as vp

base_folder = 'C:/Users/shires/DeepLabCutData/cell_01_video_3/'
video_name = 'AH0698x170601-3.mp4'


def create_video(clip):
    ny, nx, fps = clip.height(), clip.width(), clip.fps()
    n_frames = clip.frame_count()

    video = cv2.VideoWriter(os.path.join(base_folder, video_name.split('.')[0] + '-.avi'),
                            cv2.VideoWriter_fourcc(*"XVID"), fps, (nx, ny))

    for index in tqdm(range(n_frames)):
        image = clip.load_frame()

        frame = image
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    clip.close()


def make_labeled_video():
    clip = vp(os.path.join(base_folder, video_name),
              os.path.join(base_folder, video_name.split('.')[0] + '-labeled.mp4'))
    create_video(clip)


if __name__ == '__main__':
    make_labeled_video()