# food_embeddings

## Installation

## Optional: Create and source virtual environment

### Create virtual environment for this application. Note that Python3.6 is the expected Python version used for this library.
### Replace path to virtual env with environment name
`virtualenv -p /usr/bin/python3.6 <path to virtual env>`

### Activate virtual environment
`source <path to virtual env>/bin/activate`

## Install package with requirements (cd to directory containing setup.py)
`pip install -e .`


## Training embedding networks
Script: train_distance_triplet_net_audio.py (train network using audio data)
1) -thresh_filepath
Need to enter CL argument specifying the filepath containing the thresholds for the audio PCA feature data - this gives the threshold for each sample to determine the N nearest neighbors to select from when determining triplet positive and negative samples during network training. The file used in audio training is "all_sound_PCA_feat_thresh_10NNs.npy", located in the data folder.

2) -audio_labels_filename
Need to enter command line argument specifying location of the labels for audio data - these labels are vectors of PCA features from the raw audio data. Labels stored in the file are arrays of size num_samplesx6 PCs. The file used here is "audio_PCA_feature_labels_9.pkl", located in the data folder.

Script: train_distance_triplet_net_vegLabels.py (train network using food type labels)
1) -vegType_labels_filename
Need to specify location of numerical vegetable type labels. The file used here is 'vegNames_numerical.txt' located in the data folder.


## Saving embeddings from learned model
Script: save_learned_embeds_audio_triplet_net.py
CL arguments:
1) -thresh_filepath
Need to enter CL argument specifying the filepath containing the thresholds for the audio PCA feature data - this gives the threshold for each sample to determine the N nearest neighbors to select from when determining triplet positive and negative samples during network training. The file used in audio training is "all_sound_PCA_feat_thresh_10NNs.npy", located in the data folder.

2) -audio_labels_filename
Need to enter command line argument specifying location of the labels for audio data - these labels are vectors of PCA features from the raw audio data. Labels stored in the file are arrays of size num_samplesx6 PCs. The file used here is "audio_PCA_feature_labels_9.pkl", located in the data folder.

3) -saved_checkpoint 
Need to enter command line argument specifying location of the saved learned embedding model. Current checkpoint is checkpts/run12_emb16_10NNs_moreSaving/checkpoint4.pth.tar.
