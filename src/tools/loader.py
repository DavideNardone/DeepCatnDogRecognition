from __future__ import print_function

import numpy as np
import os
import process_data as pi
import cPickle as pickle

TRAIN_DIR = 'train/'
TEST_DIR = 'test/'
IMG_DIR = 'npys/'

IMG_SIZES = [64, 227]
PIXEL_DEPTH = 255.0


def label_img(img):
    word_label = img.split('.')[-3]

    if word_label == 'cat':
        return 0
    elif word_label == 'dog':
        return 1


def load_data(config):

    # read and pre-process images for each class
    train_dogs = []
    train_cats = []

    img_size = config.image_size
    model_name = config.model_name

    if (not os.path.exists(IMG_DIR + 'train_dogs' + str(img_size) + '.npy')) and (not os.path.exists(IMG_DIR + 'train_cats' + str(img_size) + '.npy')):
        print('Reading and processing train images...')
        for img in os.listdir(TRAIN_DIR):

            if model_name == 'Alex-Net':
                img = pi.process_image(TEST_DIR + img, img_size)
            else:
                img = pi.process_image(TEST_DIR + img, img_size)
                img = pi.normalize_image(img, PIXEL_DEPTH, mean_vec=None)

            label = label_img(img)

            #dog
            if(label == 1):
                train_dogs.append([np.array(img), label])
            #cat
            else:
                train_cats.append([np.array(img), label])

        np.save(IMG_DIR + 'train_dogs' + str(img_size) + '.npy', train_dogs)
        np.save(IMG_DIR + 'train_cats' + str(img_size) + '.npy', train_cats)
    else:
        print('Loading train set...')
        train_cats = np.load(IMG_DIR + 'train_cats' + str(img_size) + '.npy')
        train_dogs = np.load(IMG_DIR + 'train_dogs' + str(img_size) + '.npy')

    print('Done!')

    np.random.shuffle(train_dogs)
    np.random.shuffle(train_cats)

    return np.array(train_dogs), np.array(train_cats)


def load_test_data(config):

    # read and pre-process images for each class

    img_size = config.image_size
    model_name = config.model_name

    test_data = []
    if not os.path.exists(IMG_DIR + 'test_data' + str(img_size) + '.npy'):
        print('Reading and processing test images...')
        for img in os.listdir(TEST_DIR):
            index = img[:-4]

            if model_name == 'Alex-Net':
                img = pi.process_image(TEST_DIR + img, img_size)
            else:
                img = pi.process_image(TEST_DIR + img, img_size)
                img = pi.normalize_image(img, PIXEL_DEPTH, mean_vec=None)

            test_data.append([np.array(img), index])
        np.save(IMG_DIR + 'test_data' + str(img_size) + '.npy', test_data)
    else:
        print('Loading test set...')
        test_data = np.load(IMG_DIR + 'test_data' + str(img_size) + '.npy')

    return np.array(test_data)


def prepare_test_data(test_data):

    # split each batch in batches

    n = 100
    print("Generating test batches...")
    batches = np.array(np.array_split(test_data, n))
    return batches


def create_batches(train_dogs, train_cats, batch_size):

    # split each dataset into batches

    print("Generating training batches")
    split = batch_size / 2
    dogs = np.array(np.array_split(train_dogs, 12500 / split))
    cats = np.array(np.array_split(train_cats, 12500 / split))

    np.random.shuffle(dogs)
    np.random.shuffle(cats)

    # Choose mini batch from each
    batches = []
    for dog, cat in zip(dogs, cats):
        batch = np.concatenate([dog, cat])
        np.random.shuffle(batch)
        batches.append(batch)

    return batches


def init_data(config):

    #loading data
    train_dogs, train_cats = load_data(config)

    #retrieve batches
    train_batches = create_batches(train_dogs, train_cats, config.mini_batch_size)

    # print(len(train_batches))

    # percentage for validation set from train set
    val_size = int(len(train_batches) * config.val_size)

    # Split data set into train and validation set
    valid_batches = train_batches[-val_size:]
    train_batches = train_batches[:-val_size]

    return train_batches, valid_batches


def init_test_data(config):
    test_data = load_test_data(config)
    test_batches = prepare_test_data(test_data)
    test_data = []

    return test_batches