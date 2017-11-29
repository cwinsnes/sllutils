"""
Module for creating and working with Deep Neural Networks.
No warranty!
"""
import itertools
import tempfile
import tensorflow as tf
import utils.iterutils as iterutils
import utils.tfutils as tfutils
import collections


class DNN(object):
    """
    Implements (Deep) Neural Networks using TensorFlow layers.
    """
    def __init__(self):
        self._previous_layer = None
        self._X = None
        self._counter = collections.Counter()
        self._session = tf.Session()
        self._dropouts = []

    def set_input(self, shape):
        """
        Sets the input layer of the network.

        Args:
            shape: a list variable determining the size of the input layer.
                   For variable number of inputs, this list should begin with a 'None'.
                   For example, MNIST would use shape=[None, 784]
        """
        if self._X is not None:
            raise ValueError('Setting the input layer more than once is impossible')

        with tf.name_scope('input'):
            self._X = tf.placeholder(tf.float32, shape=shape)
            self._previous_layer = self._X

    def add_reshape(self, new_shape):
        if self._previous_layer is None:
            raise ValueError('An input layer must be set first')

        with tf.name_scope('reshape' + str(self._counter['reshape'])):
            self._previous_layer = tf.reshape(self._previous_layer, new_shape)

    def add_conv2d(self, kernel_size, filters, padding='same', activation=tf.nn.relu):
        self._previous_layer = tf.layers.conv2d(inputs=self._previous_layer,
                                                filters=filters,
                                                kernel_size=kernel_size,
                                                padding=padding,
                                                activation=activation)

    def add_max_pool2d(self, pool_size, strides):
        self._previous_layer = tf.layers.max_pooling2d(inputs=self._previous_layer,
                                                       pool_size=pool_size,
                                                       strides=strides)

    def add_dense(self, units, activation=tf.nn.relu):
        self._previous_layer = tf.layers.dense(inputs=self._previous_layer,
                                               units=units,
                                               activation=activation)

    def add_dropout(self, drop_probability):
        prob = tf.placeholder(tf.float32)
        self._previous_layer = tf.layers.dropout(self._previous_layer, rate=prob, training=True)
        self._dropouts.append((prob, drop_probability))

    def build(self, cost_function=tfutils.binary_cross_entropy, optimizer=tf.train.AdamOptimizer()):
        self._Y = tf.placeholder(tf.float32, [None, *self._previous_layer.get_shape().as_list()[1:]])
        self._predict_op = self._previous_layer
        self._cost_op = tf.reduce_mean(cost_function(self._predict_op, self._Y))
        self._train_op = optimizer.minimize(self._cost_op)
        self._session.run(tf.global_variables_initializer())

    def train(self, train_x, train_y):
        feed_dict = {self._X: train_x,
                     self._Y: train_y}
        for prob, prob_val in self._dropouts:
            feed_dict[prob] = prob_val
        self._session.run(self._train_op, feed_dict=feed_dict)

    def cost(self, x, y):
        feed_dict = {self._X: x,
                     self._Y: y}
        for prob, prob_val in self._dropouts:
            feed_dict[prob] = 0
        return(self._session.run(self._cost_op, feed_dict=feed_dict))

    def predict(self, x):
        feed_dict = {self._X: x}
        for prob, prob_val in self._dropouts:
            feed_dict[prob] = 0
        return(self._session.run(self._predict_op, feed_dict=feed_dict))
