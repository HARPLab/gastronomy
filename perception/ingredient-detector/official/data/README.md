This directory contains scripts to help run the data extraction pipeline as well as visualize model features and predictions. The data extraction pipeline is designed for the Recipe1M dataset, available for download [here](http://im2recipe.csail.mit.edu/dataset). 

# Data extraction pipeline
1. Run `get_salad_ids.py` to obtain salad recipes (names and name-to-image-path map) from Recipe1M.
2. Run `get_vocab.py` to identify all ingredients and ingredient counts from the salad recipes in step 1.
3. Run `find_top_ingrs.py` to indentiy top ingredients (based on count thresholding). Note that our manually curated list of ingredients are available in `ingrs_short_list.txt`.
4. Run `make_data_from_shortlist.py` to create train and validation directories of the shortlisted ingredients and recipes.
5. Run `make_tfrecords.py` to make TFRecord shards that can be used for training and validation from the directories created in step 5.

## Salad data extraction procedure (implemented in `py` files)
`det_ingrs.json`, `layer1.json`, `layer2.json` are in Recipe1M.

Detected ingredients are in `det_ingrs.json` (from bidirectional LSTM).

`layer1.json` provides mapping between recipe name and recipe ID.

`layer2.json` provides mapping between recipe ID and all associated dish image paths (from a tar file).

Find (10 digit) unique IDs of all salad dishes in `layer1.json`.
For each such ID,
* Find all related image paths using `layer2.json` (maps dish ID to image path).
* Find all detected ingredients using `det_ingrs.json`.

42k images - 90% training, 10% validation

10k ingredients

# Visualize predicted ingredients
To print the actual (ground truth) and predicted ingredients on to the test image, run the file `plot_pred.py`. To see how to obtain this data, see the readme file in directory `resnet`.

# Nearest neighbor search of train and validation features
To identify a set of images from train/validation based on the nearest neighbor distance of their features from those of images from validation/train respectively, 
1. Prepare `pkl` files for KDTree search in `make_viz_inputs.py`.
2. Run search in `visualize_nn_pca_nod.py`.

To obtain features for nearest neighbor search, read the readme file in directory `resnet`.
