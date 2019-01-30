This directory contains files for training a ResNet V2 (50) model on the dataset extracted as per the scripts in the `data` directory. The network is set up to perform a multi-label classification task of identifying ingredients of a given salad image.  

# Pre-requisites
Add the top-level `/ingredient-detector` folder to the Python path with the command:
```
export PYTHONPATH="$PYTHONPATH:/path/to/ingredient-detector"
```

# Training the network
We warm-start with ResNet V2 (50) network weights trained on the ImageNet classification task.
```
python imagenet_main.py --data_dir data/scratch/ --num_gpus 1 --pretrained_model_checkpoint_path [CKPT-PATH] --resnet_version 2 --model_dir [OUT-DIR] --learning_rate 0.256 --batch_size 32
```
Where `CKPT-PATH` is for example, `resnet_imagenet_v2_fp32_20181001/model.ckpt-225207` if we use the pretrained model [files](http://download.tensorflow.org/models/official/20181001_resnet/checkpoints/resnet_imagenet_v2_fp32_20181001.tar.gz) from the official TF repo.

# Get ingredient predictions and last layer features for NN search
```
python imagenet_main.py --data_dir data/scratch/ --num_gpus 1 --pretrained_model_checkpoint_path [CKPT-PATH] --resnet_version 2 --model_dir [OUT-DIR] --eval_only True
```
If you want to obtain the features and predictions for the training data, additionally set the flag `--use_train_data True`. Here `CKPT-PATH` is the path to the ckpt files of the trained model.
