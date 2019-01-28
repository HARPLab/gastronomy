import numpy as np
import pickle
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image, ImageFont, ImageDraw
from sklearn.metrics import precision_score, recall_score

# TODO(sai): print model probabilities with the text
VOCAB_PKL_PATH = 'vocab_short_list.pkl'
vocab = pickle.load(open(VOCAB_PKL_PATH, 'rb'))
if len(vocab) == 2:
    vocab = vocab[0]
    vocab = list(vocab)

vocab = np.array(vocab)
_NUM_CLASSES = 87
assert _NUM_CLASSES == len(vocab)

PRED_PKL_ROOT = '/home/syalaman/random/val'
PRED_PKL_PATH = os.path.join(PRED_PKL_ROOT, 'preds.pkl')
preds = pickle.load(open(PRED_PKL_PATH, 'rb'))

DATA_PATH = glob.glob('scratch/VALIDATION/*')
DATA_PATH = sorted(DATA_PATH)

OUT_PATH = os.path.join(PRED_PKL_ROOT, 'results')
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

#font = ImageFont.truetype('Roboto-Bold.ttf', size=25)
(x1, y1) = (10, 10)
(x2, y2) = (150, 10)
color1 = 'rgb(255, 0, 0)'
color2 = 'rgb(0, 0, 255)'

with tf.Session() as sess:
    feature_map = {
        'image/encoded': tf.FixedLenFeature([], dtype=tf.string,
            default_value=''),
        'image/ingrs': tf.VarLenFeature(dtype=tf.int64)
    }

    filename_queue = tf.train.string_input_producer(DATA_PATH, num_epochs=1, shuffle=False)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(serialized_example, features=feature_map)

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3)

    label = tf.sparse_to_indicator(features['image/ingrs'], _NUM_CLASSES)

    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    count = 0
    ground_truth = []
    predictions = [pred['classes'] for pred in preds]
    probabilities = [pred['probabilities'] for pred in preds]
    try:
        while True:
            dish, ingrs = sess.run([image, label])
            ground_truth.append(ingrs)
            #plt.imshow(dish)
            #plt.savefig(os.path.join(OUT_PATH, '%04d.jpg'%(count)))
            #plt.close()
            pred = np.array(preds[count]['classes'], dtype=bool)
            probs = np.array(preds[count]['probabilities'], dtype=float)
            #with open(os.path.join(OUT_PATH, '%04d.txt'%(count)), 'w') as f:
            #    f.write(', '.join(vocab[pred]) + '\n')
            #    f.write(', '.join(vocab[ingrs]))
            im = Image.fromarray(dish)
            draw = ImageDraw.Draw(im)
            text1 = '\n'.join('{} : {:.2f}'.format(_v, float(_p)) for _v, _p in (zip(vocab[pred], probs[pred])))
            draw.text((x1, y1), text1, fill=color1)#, font=font)
            draw.text((x2, y2), '\n'.join(vocab[ingrs]), fill=color2)#, font=font)
            im.save(os.path.join(OUT_PATH, '%04d.jpg'%(count)))

            print count
            count += 1
    except tf.errors.OutOfRangeError, e:
        coord.request_stop(e)
    finally:
        coord.request_stop()
        coord.join(threads)

ground_truth = np.array(ground_truth, dtype=np.int32)
predictions = np.array(predictions, dtype=np.int32)

P = {}
R = {}
for i, word in enumerate(vocab):
    y_true = ground_truth[:, i]
    y_pred = predictions[:, i]
    P[word] = precision_score(y_true, y_pred)
    R[word] = recall_score(y_true, y_pred)

true_f = np.mean(ground_truth, axis=0)
pred_f = np.mean(predictions, axis=0)

f = open(os.path.join(OUT_PATH, 'precision.txt'), 'w')
sorted_p = sorted([(k, v) for k, v in P.items()], key=lambda x: -x[1])
for item in sorted_p:
    f.write('{}: {}\n'.format(item[0], item[1]))
f.close()

f = open(os.path.join(OUT_PATH, 'recall.txt'), 'w')
sorted_r = sorted([(k, v) for k, v in R.items()], key=lambda x: -x[1])
for item in sorted_r:
    f.write('{}: {}\n'.format(item[0], item[1]))
f.close()

idx = np.arange(len(vocab))
plt.bar(idx, true_f)
plt.xticks(idx, vocab, rotation=45, fontsize=3)
plt.ylabel('Frequency of occurrence')
plt.title('Ground truth data')
plt.savefig(os.path.join(OUT_PATH, 'true_f.png'), dpi=500)
plt.close()

plt.bar(idx, pred_f)
plt.xticks(idx, vocab, rotation=45, fontsize=3)
plt.ylabel('Frequency of occurrence')
plt.title('Predicted data')
plt.savefig(os.path.join(OUT_PATH, 'pred_f.png'), dpi=500)
plt.close()
