import os
import tensorflow as tf


class DeepConv(object):

    def __init__(self, config, session, graph):

        self.graph = graph
        self.sess = session
        self.log_path = '/tmp/tensorboard/'
        self.config = config
        self.learning_rate = self.config.lr
        self.batch_size = self.config.batch_size
        self.image_size = self.config.image_size
        self.epochs = self.config.epochs

        # build computation graph
        self.build_graph()

    def conv_net(self, images, training):

        # initializing weights
        initializer = tf.contrib.layers.xavier_initializer()
        regularizer = tf.contrib.layers.l2_regularizer(self.config.l2)

        # Input Layer
        input_layer = tf.reshape(images, [-1,
                     self.config.image_size,
                     self.config.image_size,
                     self.config.channels])

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer,
            activation=tf.nn.relu
        )

        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=(2, 2))
        # conv1_bn = tf.layers.batch_normalization(pool1)

        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=256,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer,
            activation=tf.nn.relu)

        #max pooling
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=(2, 2))
        # conv2_bn = tf.layers.batch_normalization(pool2)

        # Convolutional Layer #3
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer,
            activation=tf.nn.relu
        )

        #max pooling
        pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=(2, 2))
        # conv3_bn = tf.layers.batch_normalization(pool3)

        # Convolutional Layer #4
        conv4 = tf.layers.conv2d(
            inputs=pool3,
            filters=1024,
            kernel_size=[3, 3],
            padding="same",
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer,
            activation=tf.nn.relu
        )

        #max pooling
        pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=(2, 2))
        # conv4_bn = tf.layers.batch_normalization(pool4)

        # Dense Layer
        flatten = tf.contrib.layers.flatten(pool4)

        # Fully Connected Layer
        fc1 = tf.layers.dense(
            inputs=flatten,
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer)

        # drop out
        fc1 = tf.layers.dropout(
            inputs=fc1,
            rate=self.config.dropout,
            training=training)

        # fc1 = tf.layers.batch_normalization(fc1)

        # Fully Connected Layer
        fc2 = tf.layers.dense(
            inputs=fc1,
            units=256,
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer)

        # drop out
        fc2 = tf.layers.dropout(
            inputs=fc2,
            rate=self.config.dropout,
            training=training)

        # fc2 = tf.layers.batch_normalization(fc2)

        # Fully Connected Layer
        fc3 = tf.layers.dense(
            inputs=fc2,
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer)

        # drop out
        fc3 = tf.layers.dropout(
            inputs=fc3,
            rate=self.config.dropout,
            training=training)

        # fc3 = tf.layers.batch_normalization(fc3)

        # Fully Connected Layer
        fc4 = tf.layers.dense(
            inputs=fc3,
            units=1024,
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer,
            use_bias=True,
            bias_initializer=initializer,
            bias_regularizer=regularizer)

        # drop out
        fc4 = tf.layers.dropout(
            inputs=fc4,
            rate=self.config.dropout,
            training=training)

        # fc4 = tf.layers.batch_normalization(fc4)

        # One output: Confidence score of being a dog
        logits = tf.layers.dense(inputs=fc4, units=1, activation=tf.nn.sigmoid)

        return logits

    # build the graph
    def build_graph(self):
        with self.graph.as_default():
            with self.sess:
                with tf.device('/gpu:0'): #using gpu0
                    # images input placeholder
                    self.images = tf.placeholder(shape=[None,
                                                        self.config.image_size,
                                                        self.config.image_size,
                                                        self.config.channels],
                                                 dtype=tf.float32,
                                                 name='X')

                    # Input labels that represent the real outputs
                    self.labels = tf.placeholder(shape=[None, 1],
                                                 dtype=tf.float32,
                                                 name='y')

                    self.training = tf.placeholder(dtype=tf.bool)

                    #initializing the model
                    self.model = self.conv_net(self.images, self.training)
                    thresholds = tf.fill([self.config.batch_size], self.config.threshold)

                    self.predictions = tf.greater_equal(self.model, thresholds)

                    correct_prediction = tf.equal(self.predictions, tf.cast(self.labels, tf.bool))

                    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

                    self.loss = tf.losses.log_loss(labels=self.labels, predictions=self.model)

                    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
                    # self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

                    # TensorBoard Summary
                    tf.summary.scalar("log_loss", self.loss)
                    tf.summary.scalar("accuracy", self.accuracy)
                    self.summary = tf.summary.merge_all()

                    self.init = tf.global_variables_initializer()
                    self.writer = tf.summary.FileWriter(self.log_path, graph=self.sess.graph_def)

                with tf.device('/cpu:0'):
                    self.saver = tf.train.Saver(tf.trainable_variables())

    def getFeedDict(self, batch_images, batch_labels, training=False):
        return {
            self.images: batch_images,
            self.labels: batch_labels,
            self.training: training
        }

    def predict(self, batch_images, batch_labels):
        feed_dict = self.getFeedDict(batch_images, batch_labels, False)
        pred, loss, acc = self.sess.run([self.model, self.loss, self.accuracy], feed_dict=feed_dict)
        return pred, loss, acc

    def train_eval_batch(self, batch_images, batch_labels, training=True):

        feed_dict = self.getFeedDict(batch_images, batch_labels, training)
        loss, acc, _ = self.sess.run([self.loss, self.accuracy, self.optimizer], feed_dict=feed_dict)

        return loss, acc

    def eval_batch(self, batch_images, batch_labels, training=False):

        feed_dict = self.getFeedDict(batch_images, batch_labels, training)
        summary, loss, acc = self.sess.run([self.summary, self.loss, self.accuracy], feed_dict=feed_dict)

        return summary, loss, acc

    def test_batch(self, batch_images, batch_labels, training=False):

        feed_dict = self.getFeedDict(batch_images, batch_labels, training)
        pred = self.sess.run([self.model], feed_dict=feed_dict)

        return pred

    def save(self, epoch):

        if not os.path.isdir(self.config.ckpt_path):
            os.mkdir(self.config.ckpt_path)

        self.saver.save(self.sess, self.config.ckpt_path + '/' + self.config.model_name + '.ckpt', global_step=epoch)

    def restore(self, path=None):
        # get checkpoint state
        if path:
            ckpt = tf.train.get_checkpoint_state(path)
        else:
            ckpt = tf.train.get_checkpoint_state(self.config.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            self.sess.run(self.init)
            print("\nGlobal Variables Initialized")
            self.saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
            print("\nRestoring model")
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            self.sess.run(self.init)

        print("\nGlobal Variables Initialized")