# dlc-cloudml
Port to cloud-ml for tensorflow parts of DeepLabCut

This is an adaptation of DeepLabCut: https://github.com/AlexEMG/DeepLabCut. All code is adapted from this original repository and this guide is also based on the documentation found there. I recommend checking out the original repo as well as the paper: rdcu.be/4Rep before using this code.

The code here is intended to be used where continuous access to a local GPU is not possible, CPU training is not feasible and use of a cloud VM is prohibitively expensive. Instead, computationally intensive parts of the DeepLabCut pipeline (model training) are shifted to Google cloud computing services through its job system. 

This pipeline uses the paid services of Google Cloud, you will need to enable billing and potentially spend money for the cloud based portion of this guide. New accounts get a certain amount of computing time free in the trial period and this pipeline should not exceed this allowance, but do keep in mind that you may eventually be billed by Google for multiple runs. Check the Getting Started Guide for more detailed info: https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction 

## Installation
Clone the repo locally and install the dependencies specified in "requirements.txt" in the root folder. You'll need a Google Cloud Platform account as well.

Note: tested on Windows only

## Usage - Preprocessing
### Data format
This guide assumes you have all your video data that you want to track in a single folder. This pipeline will generate all files needed for training in this folder so you only ever have to set a single root folder (in theory).

### Download pretrained weights
Inside dlc-cloudml, navigate to models/pretrained and run "download.sh" to get the pretrained resnet weights. You should end up with two .ckpt files within models/pretrained containing the resnet weights.

### Selecting random frames for the training dataset
Inside dlc-cloudml, navigate to the local_processing folder and run "select_random_frames.py". You will need to edit some of the parameters at the top of the script to work with your own data:
- task = < experiment name >
- video_path = < path to your video data folder >
- video_type = < file extension for your videos, e.g. .mp4 >

The other parameters - e.g. date, cropping parameters - don't necessarily need to be changed but you may find it useful for keeping track of different model runs.
Once the script has finished running there should be a new folder within your video data folder named after your task parameter. Within this new folder should be another folder containing the randomly selected images for training: < parent folder > / < data-task > / selected

### Manually label the training data
Next you should manually label the features of interest as specified in step 2 of the DeepLabCut guide: https://github.com/AlexEMG/DeepLabCut/blob/master/docs/demo-guide.md

Save the markers for each feature as a separate .csv file in the < parent folder > / < data-task > / selected / path generated in the previous step. (i.e. your marker location .csv files live in the same folder as your selected images)

### Generate a label dataset
In the local_processing folder of dlc-cloudml, run "labels_to_data_frame.py". Make sure to set the parameters according to your own data:
- task = < experiment name > (same as in random frame selection)
- base_folder = < path to your video data folder > (same as in random frame selection)
- scorer = < your name >
- image_type = < image format >
- body_parts = [< each >, < labeled >, < feature >] (names should correspond to the names of the .csv files you generated in the previous step)

You should now have have files names collected-data-< scorer > in < parent folder > / < data-task >, which contain all the manually labeled marker data together.

### Check your labels
In the local_processing folder of dlc-cloudml, run "check_labels.py". Again, make sure the parameters in this script are consistent with the previous steps. Once this script has finished, you should have a folder named "selected-labeled" along with the "selected" folder. Look through the images to make sure the labels are consistent with the features you want to track.

### Generate training files
In the local_processing folder of dlc-cloudml, run "generate_training_file.py". Once again make sure relevant parameters are consistent with the previous steps. Also set:
- shuffles = [< number of data shuffles >]
- training_fraction [ < how to split training and test data between each shuffle > ] (e.g. 0.95 = 95% training data, 5% test data)

Note that this step will create the actual data that will be sent to the cloud as well as the configuration files used to train the model. Within "generate_training_file.py" is the function "generate_base_config()" which contains a dictionary specifying training parameters. You can also change these parameters before running this script to define how the model will train.

Once you've run this step you should have two new folders in your video data directory. One will be named something like "< task - date - params>" and contain train and test .yaml files for configuration. The other will have the prefix "unaugmented-data-set". There should also be a base "pose_cfg.yaml" in the video data directory. 

## Usage - Cloud Training
### Setting up the environment
In a web broswer, navigate to the Google Cloud Platform: https://console.cloud.google.com. Create a new project, e.g. "dlc-test" for the purposes of this guide. Make sure you enable billing and APIs for the created project: https://cloud.google.com/ml-engine/docs/tensorflow/getting-started-training-prediction

Acivate the cloud shell (top right of project dashboard) to bring up the command line interface. On the command line, clone this repository to your Google Cloud environment:

    $ git clone https://github.com/RoboDoig/dlc-cloudml
    
Then navigate into the repo:

    $ cd dlc-cloudml
    
Create a storage bucket to hold the data for training the model:

    $ PROJECT_ID=$(gcloud config list project --format "value(core.project)")
    $ BUCKET="${PROJECT_ID}-ml"
    $ gsutil mb -c regional -l us-central1 gs://${BUCKET}
    
You should now have storage bucket created, check by going to: https://console.cloud.google.com/storage/browser and selecting your project. Navigate into the bucket, for this example it should be called "dlc-test-ml".

Copy the "models" folder of your local repo containing the .ckpt resnet weights into "dlc-test-ml". Create a new folder in "dlc-test-ml", this will contain the training data, for this guide we'll call it "test_model". 
Within the bucket your file structure should now be:
  - dlc-test-ml
    - models
       - pretrained
          - download.sh
          - resnet_v1_50.ckpt
          - resnet_v1_101.ckpt
    - test_model

Copy the two folders created in the "generate training files" step (one should have "trainset" in the name, the other should have "unaugmented-data-set") into dlc-test-ml/test-model. 

### Running the training
Go back to the dashboard command line shell and create a name for the training job, I usually just use a date and time naming system:

    $ JOB_NAME="job_$(date +%Y%m%d_%H%M%S)"
  
Now submit the training job:

    $ gcloud ml-engine jobs submit training ${JOB_NAME} \
    $  --package-path trainer \
    $  --module-name trainer.task \ 
    $  --staging-bucket gs://${BUCKET} \ 
    $  --job-dir gs://${BUCKET}/${JOB_NAME} \ 
    $  --runtime-version 1.9 \
    $  --region us-central1 \ 
    $  --config job_config/config.yaml \ 
    $  -- \
    $  --data_dir gs://${BUCKET}/test_model \
    $  --weights_dir gs://${BUCKET}/models \
    $  --log_dir gs://${BUCKET}/${JOB_NAME}

If the training job has submitted successfully, you should be able to see it in the jobs viewer: https://console.cloud.google.com/mlengine/jobs. Click on "View Logs" to keep track of the job and see if any errors pop up. 
Once the job is running, you should see the iteration number, loss, and learning rate pop up periodically. Back in the command line shell, you can also monitor training progress with TensorBoard with the command:

    $ tensorboard --port 8080 --logdir gs://${BUCKET}/${JOB_NAME}
    
Then clicking "Web preview" --> "Preview on port 8080" in the top right of the shell window.

If you've made a mistake with your job parameters and the training doesn't crash out, you can cancel the job with:

    $ gcloud ml-engine jobs cancel ${JOB_NAME}
    
Note that the scale tier used by the cloud service (e.g. CPU / GPU) is defined in job_config/config.yaml in this repo. Adjust the "scaleTier" field to specify how the job will run. You will probably want to use "BASIC_GPU" in most cases for fast training, but for testing this code out you can just use "BASIC" so as not to incur the higher costs of GPU training on the cloud.
    
## Usage - Evaluation and Analysis
### Getting your trained model

As the model runs, snapshots will ocassionaly be stored in the job folder of your storage bucket on Google Cloud. In your video data directory (on local machine) create a folder called "trained-results". In the cloud job folder download 4 files and store them in the "trained-results" folder locally:

- events file (prefix: events.out.tfevents...)
- snapshot-< iteration >.data
- snapshot-< iteration >.index
- snapshot-< iteration >.meta
    
### Evaluate and analyse the model
In local_processing, run "evaluate_model.py", making sure parameters are consistent with previous steps. This will generate a new folder within "trained-results" named "evaluation" which contains the evaluation results.

Next run "analyse_results.py" which will print the training and test error and generate a set of labeled images in "trained-results/evaluation/labeled" showing the original ground truth labels overlaid with the model prediction.

### Track an entire video
local_processing/video_maker contains tools for running model inference on an entire video. First run "analyse_video.py" with the same parameters as in previous steps. Change video name to refer the the video in the video data directory you want to to track (e.g. "video_1.mp4"). Next run "make_labeled_video.py" with the same parameters, also setting:

- resnet = < int id of the resnet used >
- snapshot = < int id of the iteration of the downloaded snapshot >

If this works, you should end up with a new video with the suffix "labeled" overlaid with the model predicted positions of the tracked features.
