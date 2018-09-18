import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.color
from skimage import io
import os
from trainer.dataset.pose_dataset import data_to_input
import trainer.predict as predict


def get_cmap(n, name='cool'):
    return plt.cm.get_cmap(name, n)


def pairwise_distances(data_combined, scorer1, scorer2, pcutoff=-1, bodyparts=None):
    mask = data_combined[scorer2].xs('likelihood', level=1, axis=1) >= pcutoff
    if bodyparts is None:
        point_wise_squared_distance = (data_combined[scorer1] - data_combined[scorer2])**2
        mse = np.sqrt(point_wise_squared_distance.xs('x', level=1, axis=1)
                      + point_wise_squared_distance.xs('y', level=1, axis=1))
        return mse, mse[mask]
    else:
        point_wise_squared_distance = (data_combined[scorer1][bodyparts] - data_combined[scorer2][bodyparts])**2
        mse = np.sqrt(point_wise_squared_distance.xs('x', level=1, axis=1)
                      + point_wise_squared_distance.xs('y', level=1, axis=1))
        return mse, mse[mask]


def make_labeled_image(data_combined, imagenr, image_file_name, scorers, bodyparts, colors, labels=['+', '.', 'x'],
                       scaling=1, alphavalue=.5, dotsize=15, pcutoff=-1):
    plt.axis('off')
    im = io.imread(os.path.join(image_file_name, data_combined.index[imagenr]))
    if np.ndim(im) > 2:
        h, w, numcolors = np.shape(im)
    else:
        h, w = np.shape(im)

    plt.figure(frameon=False, figsize=(w*1./100*scaling, h*1./100*scaling))
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.imshow(im, 'gray')

    for scorer_index, loop_scorer in enumerate(scorers):
        for bp_index, bp in enumerate(bodyparts):
            if np.isfinite(data_combined[loop_scorer][bp]['y'][imagenr] + data_combined[loop_scorer][bp]['x'][imagenr]):
                y, x = int(data_combined[loop_scorer][bp]['y'][imagenr]), int(data_combined[loop_scorer][bp]['x'][imagenr])
                if 'deep-cut' in loop_scorer:
                    p = data_combined[loop_scorer][bp]['likelihood'][imagenr]
                    if p > pcutoff:
                        plt.plot(x, y, labels[1], ms=dotsize, alpha=alphavalue, color=colors(int(bp_index)))
                    else:
                        plt.plot(x, y, labels[2], ms=dotsize, alpha=alphavalue, color=colors(int(bp_index)))
                else:
                    plt.plot(x, y, labels[0], ms=dotsize, alpha=alphavalue, color=colors(int(bp_index)))

    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.axis('off')
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    plt.gca().invert_yaxis()
    return 0


def get_pose(image, cfg, inputs, outputs, sess, outall=False):
    image_batch = data_to_input(skimage.color.gray2rgb(image))
    outputs_np = sess.run(outputs, feed_dict={inputs: image_batch})
    scmap, locref = predict.extract_cnn_output(outputs_np, cfg)
    pose = predict.argmax_pose_predict(scmap, locref, cfg.stride)
    if outall:
        return scmap, locref, pose
    else:
        return pose