"""
DeepLabCut Toolbox
https://github.com/AlexEMG/DeepLabCut
A Mathis, alexander.mathis@bethgelab.org
M Mathis, mackenzie@post.harvard.edu

"""
import os
import pickle
import pandas as pd


def attempt_to_make_folder(folder_name):
    if os.path.isdir(folder_name):
        print("Folder already exists!")
    else:
        os.mkdir(folder_name)


def save_data(predict_data, meta_data, data_name, pd_index, image_names):
    DataMachine = pd.DataFrame(predict_data, columns=pd_index, index=image_names)
    DataMachine.to_hdf(data_name, 'df_with_missing', format='table', mode='w')
    with open(data_name.split('.')[0] + 'includingmetadata.pickle', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(meta_data, f, pickle.HIGHEST_PROTOCOL)


def get_immediate_subdirectories(a_dir):
    # https://stackoverflow.com/questions/800197/how-to-get-all-of-the-immediate-subdirectories-in-python
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def list_files_of_a_particular_type_in_folder(a_dir, a_file_type):
    return [
        name for name in os.listdir(a_dir)
        if a_file_type in name]


def get_video_list(filename, video_path, video_type):
    videos = list_files_of_a_particular_type_in_folder(video_path, video_type)
    if filename == 'all':
        return videos
    else:
        if filename in videos:
            videos = [filename]
        else:
            videos = []
            print("Video not found!", filename)
    return videos
