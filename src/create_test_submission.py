import sys
import numpy as np
import tensorflow as tf

import csv

import matplotlib
matplotlib.use('Agg')

from lenet import LeNet
from deepconv import DeepConv
from alexnet import AlexNet

from configs import Config
from tools.loader import init_data, init_test_data

TYPE_OF_MODEL = sys.argv[1]
print("Testing with " + TYPE_OF_MODEL + " CNN Model")

TEST_DIR = '/media/data/davidenardone/dataset/DOG_AND_CAT/test'

# Process test data and create batches in memory
graph = tf.Graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess_config = tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=True, gpu_options=gpu_options)
sess_config.gpu_options.allow_growth = True
sess = tf.Session(config=sess_config)

if TYPE_OF_MODEL == 'LE-NET':
    config = Config(TYPE_OF_MODEL)
    model = LeNet(config, sess, graph)
elif TYPE_OF_MODEL == 'DEEP-CONVNET':
    config = Config(TYPE_OF_MODEL)
    model = DeepConv(config, sess, graph)
elif TYPE_OF_MODEL == 'ALEX-NET':
    config = Config(TYPE_OF_MODEL)
    model = AlexNet(config, sess, graph)

model.restore()
print(TYPE_OF_MODEL + " CNN Model Restored")

test_batches = init_test_data(model.config)
print(len(test_batches))

# Get the predictions and write them into a CSV file
with open('/home/davidenardone/TENSORFLOW/Cat-vs-Dog-Tensorflow-CNN/results/' + TYPE_OF_MODEL + '.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'label'])

    for test_batch in test_batches:
        images, labels = map(list, zip(*test_batch))
        labels = np.array(labels).reshape(-1, 1)
        pred = np.array(model.test_batch(images, labels))
        print(pred.flatten().shape)
        print(labels.flatten().shape)
        for id, label in zip(labels.flatten(), pred.flatten()):
            print(int(id), label)
            writer.writerow([int(id), label])
    print('DONE!')