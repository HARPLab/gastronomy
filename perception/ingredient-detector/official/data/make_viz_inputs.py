import numpy as np
import pickle
import glob
import os
from PIL import Image

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
TRAIN_PATH = '/home/syalaman/tmp4/train'
VAL_PATH = '/home/syalaman/tmp4/val'

train_features_files = sorted(glob.glob(os.path.join(TRAIN_PATH, 'features*')))
val_features_files = sorted(glob.glob(os.path.join(VAL_PATH, 'features*')))

IMG_DIR = 'images'
TRAIN_IMG_PATH = os.path.join(TRAIN_PATH, IMG_DIR)
VAL_IMG_PATH = os.path.join(VAL_PATH, IMG_DIR)
if not os.path.exists(TRAIN_IMG_PATH):
    os.makedirs(TRAIN_IMG_PATH)
if not os.path.exists(VAL_IMG_PATH):
    os.makedirs(VAL_IMG_PATH)

train_images = []
train_features = []
train_preds = []
train_labels = []
idx = 0
image_name = '%05d.png'
for _file in train_features_files:
    print("Processing training file %d/%d"%(idx/32, len(train_features_files)))
    pickled_features = pickle.load(open(_file, 'rb'))
    images, features, preds, labels = pickled_features

    images[:, :, :, 0] += _R_MEAN
    images[:, :, :, 1] += _G_MEAN
    images[:, :, :, 2] += _B_MEAN
    images = np.array(images, dtype=np.uint8)
    for image in images:
        image_path = os.path.join(TRAIN_IMG_PATH, image_name%idx)
        train_images.append(image_path)
        im = Image.fromarray(image)
        im.save(image_path)
        idx += 1

    features = np.transpose(features, (0, 2, 3, 1))
    train_features.extend(list(features))

    train_preds.extend(list(preds))

    train_labels.extend(list(labels))

np.save(os.path.join(TRAIN_PATH, 'features'), train_features)
pickle.dump(train_images, open(os.path.join(TRAIN_PATH, 'image_files.pkl'), 'wb'))
pickle.dump([train_preds, train_labels], open(os.path.join(TRAIN_PATH, 'preds_and_labels.pkl'), 'wb'))

val_images = []
val_features = []
val_preds = []
val_labels = []
idx = 0
image_name = '%05d.png'
for _file in val_features_files:
    print("Processing validation file %d/%d"%(idx/32, len(val_features_files)))
    pickled_features = pickle.load(open(_file, 'rb'))
    images, features, preds, labels = pickled_features

    images[:, :, :, 0] += _R_MEAN
    images[:, :, :, 1] += _G_MEAN
    images[:, :, :, 2] += _B_MEAN
    images = np.array(images, dtype=np.uint8)
    for image in images:
        image_path = os.path.join(VAL_IMG_PATH, image_name%idx)
        val_images.append(image_path)
        im = Image.fromarray(image)
        im.save(image_path)
        idx += 1

    features = np.transpose(features, (0, 2, 3, 1))
    val_features.extend(list(features))

    val_preds.extend(list(preds))

    val_labels.extend(list(labels))

np.save(os.path.join(VAL_PATH, 'features'), val_features)
pickle.dump(val_images, open(os.path.join(VAL_PATH, 'image_files.pkl'), 'wb'))
pickle.dump([val_preds, val_labels], open(os.path.join(VAL_PATH, 'preds_and_labels.pkl'), 'wb'))
