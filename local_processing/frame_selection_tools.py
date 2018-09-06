#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut

A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

auxiliary functions for selecting frames uid or by (image-based) clustering.
"""

import numpy as np
import math
from skimage import io
from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm


def uniform_frames(clip, num_frames_to_pick, start, stop, index="all"):
    ''' Temporally uniformly sampling frames in interval (start,stop).
    Visual information of video is irrelevant for this method. This code is fast and sufficient (to extract distinct frames),
    when behavioral videos naturally covers many states.

    The variable Index allows to pass on a subindex for the frames.
    '''
    print("Uniformly extracting of frames from", start * clip.duration, " seconds to", stop * clip.duration,
          " seconds.")

    if index == "all":
        if start == 0:
            frames2pick = np.random.choice(math.ceil(clip.duration * clip.fps * stop), size=num_frames_to_pick - 1,
                                           replace=False)
        else:
            frames2pick = np.random.choice(
                range(math.floor(start * clip.duration * clip.fps), math.ceil(clip.duration * clip.fps * stop)),
                size=num_frames_to_pick - 1, replace=False)
        return frames2pick
    else:
        startindex = int(np.floor(clip.fps * clip.duration * start))
        stopindex = int(np.ceil(clip.fps * clip.duration * stop))
        index = np.array(index, dtype=np.int)
        index = index[(index > startindex) * (index < stopindex)]  # crop to range!
        if len(index) >= num_frames_to_pick - 1:
            return list(np.random.permutation(index)[:num_frames_to_pick - 1])
        else:
            return list(index)


def k_means_based_frame_selection(clip, num_frames_to_pick, start, stop, index="all", resize_width=30, batch_size=100,
                                  max_iter=50):
    ''' This code downsamples the video to a width of resizewidth.

    The video is extracted as a numpy array, which is then clustered with kmeans, whereby each frames is treated as a vector.
    Frames from different clusters are then selected for labeling. This procedure makes sure that the frames "look different",
    i.e. different postures etc. On large videos this code is slow.

    Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior.

    Note: this method can return fewer images than numframes2pick-1.'''

    print("Kmeans-quantization based extracting of frames from", start * clip.duration, " seconds to",
          stop * clip.duration, " seconds.")
    startindex = int(np.floor(clip.fps * clip.duration * start))
    stopindex = int(np.ceil(clip.fps * clip.duration * stop))

    if index == "all":
        index = np.arange(stopindex - startindex) + startindex
    else:
        index = np.array(index)
        index = index[(index > startindex) * (index < stopindex)]  # crop to range!

    nframes = len(index)
    if batch_size > nframes:
        batch_size = int(nframes / 2)

    if len(index) >= num_frames_to_pick - 1:
        clipresized = clip.resize(width=resize_width)
        ny, nx = clipresized.size

        DATA = np.zeros((nframes, nx, ny))
        frame0 = img_as_ubyte(clip.get_frame(0))
        if np.ndim(frame0) == 3:
            ncolors = np.shape(frame0)[2]
        else:
            ncolors = 1

        print("Extracting and downsampling...", nframes, " frames from the video.")
        for counter, index in tqdm(enumerate(index)):
            if ncolors == 1:
                DATA[counter, :, :] = img_as_ubyte(clipresized.get_frame(index * 1. / clipresized.fps))
            else:  # attention: averages over color channels to keep size small / perhaps you want to use color information?
                DATA[counter, :, :] = img_as_ubyte(
                    np.array(np.mean(clipresized.get_frame(index * 1. / clipresized.fps), 2), dtype=np.uint8))

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(n_clusters=num_frames_to_pick - 1, tol=1e-3, batch_size=batch_size, max_iter=max_iter)
        kmeans.fit(data)
        frames2pick = []
        for clusterid in range(num_frames_to_pick - 1):  # pick one frame per cluster
            clusterids = np.where(clusterid == kmeans.labels_)[0]

            numimagesofcluster = len(clusterids)
            if numimagesofcluster > 0:
                frames2pick.append(index[clusterids[np.random.randint(numimagesofcluster)]])

        return list(np.array(frames2pick))
    else:
        return list(index)
