import tensorflow as tf
import numpy as np
import os

class AlexNet(object):
    def __init__(self, config, session, graph):

        self.graph = graph
        self.sess = session
        self.config = config
        self.learning_rate = self.config.lr
        self.batch_size = self.config.batch_size
        self.image_size = self.config.image_size
        self.epochs = self.config.epochs

        self.train_layers = self.config.train_layers
        self.weights_path = self.config.weights_path
        self.dropout = self.config.dropout

        self.log_path = '/tmp/tensorboard/'

        # Call the create function to build the computational graph of AlexNet
        self.build_graph()

    def alex_conv_net(self):

        # 1st Layer: Conv (w ReLu) -> Pool -> Lrn
        conv1 = conv(self.images, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        pool1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(pool1, 2, 2e-05, 0.75, name='norm1')

        # 2nd Layer: Conv (w ReLu) -> Pool -> Lrn with 2 groups
        conv2 = conv(norm1, 5, 5, 256, 1, 1, groups=2, name='conv2')
        pool2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(pool2, 2, 2e-05, 0.75, name='norm2')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv(conv3, 3, 3, 384, 1, 1, groups=2, name='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv(conv4, 3, 3, 256, 1, 1, groups=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        dropout6 = dropout(fc6, self.dropout)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        dropout7 = dropout(fc7, self.dropout)

        # 8th Layer: FC and return unscaled activations (for tf.nn.softmax_cross_entropy_with_logits)
        self.fc8 = tf.nn.sigmoid(fc(dropout7, 4096, 1, relu=False, name='fc8'))

        return self.fc8

    def build_graph(self):
        with self.graph.as_default():
            with self.sess:
                with tf.device('/gpu:0'): #using gpu0
                    self.images = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
                    self.labels = tf.placeholder(tf.float32, shape=[None, 1])
                    self.keep_prob = tf.placeholder(tf.float32)

                    self.training = tf.placeholder(dtype=tf.bool)

                    self.model = self.alex_conv_net()

                    self.var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in self.config.train_layers]

                    # with tf.name_scope("cross_ent"):
                    self.loss = tf.losses.log_loss(labels=self.labels, predictions=self.model)

                    # with tf.name_scope("train"):        # Get gradients of all trainable variables
                    gradients = tf.gradients(self.loss, self.var_list)
                    gradients = list(zip(gradients, self.var_list))

                    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01).apply_gradients(grads_and_vars=gradients)

                    # with tf.name_scope("accuracy"):
                    thresholds = tf.fill([self.config.batch_size], self.config.threshold)
                    self.predictions = tf.greater_equal(self.model, thresholds)
                    correct_pred = tf.equal(self.predictions, tf.cast(self.labels, tf.bool))
                    self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

                    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

                    # TensorBoard Summary
                    tf.summary.scalar("log_loss", self.loss)
                    tf.summary.scalar("accuracy", self.accuracy)
                    self.summary = tf.summary.merge_all()

                    self.init = tf.global_variables_initializer()
                    self.writer = tf.summary.FileWriter(self.log_path, graph=self.sess.graph_def)

                with tf.device('/cpu:0'):
                    self.saver = tf.train.Saver(tf.trainable_variables())

    def load_initial_weights(self, session):
        """
        As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come
        as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of
        dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
        need a special load function
        """

        # Load the weights into memory
        weights_dict = np.load(self.weights_path, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if the layer is one of the layers that should be reinitialized
            if op_name not in self.train_layers:

                with tf.variable_scope(op_name, reuse=True):

                    # Loop over list of weights/biases and assign them to their corresponding tf variable
                    for data in weights_dict[op_name]:

                        # Biases
                        if len(data.shape) == 1:

                            var = tf.get_variable('biases', trainable=False)
                            session.run(var.assign(data))

                        # Weights
                        else:

                            var = tf.get_variable('weights', trainable=False)
                            session.run(var.assign(data))

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

        #create dir if it does not exist
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
            self.load_initial_weights(self.sess)

        print("\nGlobal Variables Initialized")

"""
Predefine all necessary layer for the AlexNet
"""

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding='SAME', groups=1):
    """
    Adapted from: https://github.com/ethereon/caffe-tensorflow
    """
    # Get number of input channels
    input_channels = int(x.get_shape()[-1])

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=[1, stride_y, stride_x, 1],
                                         padding=padding)

    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[filter_height, filter_width, input_channels / groups, num_filters])
        biases = tf.get_variable('biases', shape=[num_filters])

        if groups == 1:
            conv = convolve(x, weights)

        # In the cases of multiple groups, split inputs & weights and
        else:
            # Split input and weights and convolve them separately
            input_groups = tf.split(axis=3, num_or_size_splits=groups, value=x)
            weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=weights)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

            # Concat the convolved output together again
            conv = tf.concat(axis=3, values=output_groups)

        # Add biases
        bias = tf.reshape(tf.nn.bias_add(conv, biases), tf.shape(conv))

        # Apply relu function
        relu = tf.nn.relu(bias, name=scope.name)

        return relu


def fc(x, num_in, num_out, name, relu=True):
    with tf.variable_scope(name) as scope:

        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out], trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

        if relu == True:
            # Apply ReLu non linearity
            relu = tf.nn.relu(act)
            return relu
        else:
            return act


def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='SAME'):
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
                          strides=[1, stride_y, stride_x, 1],
                          padding=padding, name=name)


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)