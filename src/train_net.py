from __future__ import print_function

import os

import sys
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lenet import LeNet
from deepconvnet import DeepConv
from alexnet import AlexNet

from configs import Config
from tools.loader import init_data

graph_dir = 'graphs/'



def save_plot_files(model_name, steps, loss, acc, is_training=True):

    if is_training == True:

        title1 = 'training_loss.png'
        title2 = 'training_acc.png'
        x_label = '#steps'
        y1_label = 'Training Loss'
        y2_label = 'Training Acc'

        # saving loss
        plt.clf()
        plt.plot(steps, loss)
        plt.xlabel(x_label)
        plt.ylabel(y1_label)
        plt.savefig(graph_dir + '/' + model_name + '/' + title1)

        # saving accuracy
        plt.clf()
        plt.plot(steps, acc)
        plt.xlabel(x_label)
        plt.ylabel(y2_label)
        plt.savefig(graph_dir + '/' + model_name + '/' + title2)

    else:

        title1 = 'validation_loss.png'
        title2 = 'validation_acc.png'
        x_label = '#steps'
        y1_label = 'Validation Loss'
        y2_label = 'Validation Acc'

        # saving loss
        plt.clf()
        plt.plot(steps, loss)
        plt.xlabel(x_label)
        plt.ylabel(y1_label)
        plt.savefig(graph_dir + '/' + model_name + '/' + title1)

        # saving accuracy
        plt.clf()
        plt.plot(steps, acc)
        plt.xlabel(x_label)
        plt.ylabel(y2_label)
        plt.savefig(graph_dir + '/' + model_name + '/' + title2)


def run(model, train_batches, valid_batches):

    # For Plots
    steps_train = []
    steps_val = []
    y_training_loss = []
    y_training_accuracy = []
    y_valid_loss = []
    y_valid_accuracy = []

    print('Training the model')

    # create grap dir for each model
    if not os.path.isdir(graph_dir + '/' + model.config.model_name):
        os.mkdir(graph_dir + '/' + model.config.model_name)

    step_train = 1
    step_val = 1
    try:
        for epoch in range(model.epochs):

            train_acc = 0
            train_loss = 0

            np.random.shuffle(train_batches)

            # loop on the batches
            for train_batch in train_batches:

                train_batch_images, train_batch_labels = map(list, zip(*train_batch))

                train_batch_images = np.array(train_batch_images)
                train_batch_labels = np.array(train_batch_labels).reshape(-1, 1)

                batch_train_loss, batch_train_acc = model.train_eval_batch(train_batch_images, train_batch_labels, True)

                # print("loss: %.9f, acc %.9f" % (batch_train_loss, batch_train_acc))

                train_loss += batch_train_loss
                train_acc += batch_train_acc

                #plot variables
                y_training_loss.append(batch_train_loss)
                y_training_accuracy.append(batch_train_acc)
                steps_train.append(step_train)
                step_train += 1

            #saving loss and acc on train batches
            save_plot_files(model.config.model_name, steps_train, y_training_loss, y_training_accuracy, True)

            avg_train_loss = train_loss/len(train_batches)
            avg_train_acc = train_acc/len(train_batches)

            print('Epoch: %d, Train Loss: %f, Train  Acc: %f' % (epoch + 1, avg_train_loss, avg_train_acc))


            # validate the model after every epoch
            np.random.shuffle(valid_batches)
            val_acc = 0
            val_loss = 0
            for val_batch in valid_batches:
                valid_batch_images, valid_batch_labels = map(list, zip(*val_batch))
                valid_batch_images = np.array(valid_batch_images)
                valid_batch_labels = np.array(valid_batch_labels).reshape(-1, 1)

                summary, batch_val_loss, batch_val_acc = model.eval_batch(valid_batch_images, valid_batch_labels)
                val_acc += batch_val_acc
                val_loss += batch_val_loss

                # plot variables
                y_valid_loss.append(batch_val_loss)
                y_valid_accuracy.append(batch_val_acc)
                steps_val.append(step_val)
                step_val += 1

            #saving loss and acc on val batches
            save_plot_files(model.config.model_name, steps_val, y_valid_loss, y_valid_accuracy, False)

            avg_val_loss = val_loss/len(valid_batches)
            avg_val_acc = val_acc/len(valid_batches)

            print('Epoch: %d, Valid Loss: %f, Valid Acc: %f' % (epoch + 1, avg_val_loss, avg_val_acc))

            # Save model after every epoch
            print('saving checkpoint')
            model.save(epoch)

            #storing tensorboard results after every epoch
            model.writer.add_summary(summary, epoch)


    except KeyboardInterrupt:
        print ('Training interrupted!')
        

if __name__ == '__main__':

    model_type = sys.argv[1]

    # Initialize model
    graph = tf.Graph()
    sess = tf.Session()

    if model_type == 'LE-NET':
        config = Config(model_type)
        model = LeNet(config, sess, graph)
    elif model_type == 'DEEP-CONVNET':
        config = Config(model_type)
        model = DeepConv(config, sess, graph)
    elif model_type == 'ALEX-NET':
        config = Config(model_type)
        model = AlexNet(config, sess, graph)

    train_batches, valid_batches = init_data(model.config)

    model.restore()

    print("Training with " + model_type + " CNN Model")
    run(model, train_batches, valid_batches)
    print("Training Complete")
    model.writer.close()
