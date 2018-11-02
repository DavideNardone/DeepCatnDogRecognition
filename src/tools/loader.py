from __future__ import print_function

import numpy as np    		# dealing with arrays
import os              		# dealing with directories
import sys
import matplotlib.pyplot as plt
import cv2
import process_data as pi

TRAIN_DIR = '/media/data/davidenardone/dataset/DOG_AND_CAT/train/'
TEST_DIR = '/media/data/davidenardone/dataset/DOG_AND_CAT/test/'
IMG_DIR = '/media/data/davidenardone/dataset/DOG_AND_CAT/npys/'
BATCH_DIR = '/media/data/davidenardone/dataset/DOG_AND_CAT/batches/'

IMG_SIZES = [32, 64, 150]
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
                img = pi.process_image(TRAIN_DIR + img, img_size, PIXEL_DEPTH, mean_vec=config.MEAN)
            else:
                img = pi.process_image(TRAIN_DIR + img, img_size, PIXEL_DEPTH, mean_vec=None)
                img = pi.normalize_image(img, pixel_depth)

            label = label_img(img)

            # img = cv2.imread(TRAIN_DIR + img)
            # flip image at random if flag is selected
            # img = cv2.flip(img, 1)
            # rescale image
            # img = cv2.resize(img, (img_size, img_size))
            # img = img.astype(np.float32)
            # # subtract mean
            # img -= MEAN

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
        for test in os.listdir(TEST_DIR):
            index = test[:-4]

            if model_name == 'Alex-Net':
                img = pi.process_image(TEST_DIR + img, img_size, PIXEL_DEPTH, mean_vec=config.MEAN)
            else:
                img = pi.process_image(TEST_DIR + img, img_size, PIXEL_DEPTH, mean_vec=None)
                img = pi.normalize_image(img, pixel_depth)

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


def split_batches(batches, mini_batch_size):

    #split each batch in mini-batch

    batches = np.array_split(batches, mini_batch_size)

    for i in range(mini_batch_size):
        np.save(BATCH_DIR + 'batch' + str(i) + '.npy', batches[i])


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

if __name__ == '__main__':
    for img_size in IMG_SIZES:
        train_dogs, train_cats = load_data(img_size)
        print("Train Dogs: {}-{}".format(train_dogs.shape,
                                         train_dogs[0][0].shape))
        print("Train Cats: {}-{}".format(train_cats.shape,
                                         train_cats[0][0].shape))
    # train_dogs, train_cats = load_data(64)
    # print("Train Dogs: {}".format(train_dogs.shape))
    # print("Train Cats: {}".format(train_cats.shape))
    # print(train_cats[0].shape)
    # print(train_cats[0][0].shape)
    # print(train_cats[0][1])
    # batches = prepare_train_data(train_dogs, train_cats, 32)
    # print(len(batches))
    # split_batches(batches)

    # plt.imshow(train_dogs[0][0], cmap='gray')
    # plt.show()
    # plt.imshow(train_dogs[1][0], interpolation='nearest')
    # plt.figure()
    # plt.imshow(train_dogs[2][0], interpolation='nearest')
    # plt.figure()
    # plt.imshow(train_cats[0][0], interpolation='nearest')
    # plt.figure()
    # plt.imshow(train_cats[1][0], interpolation='nearest')
    # plt.figure()
    # plt.imshow(train_cats[2][0], interpolation='nearest')
    # plt.figure()
    # prepare_train_data()
