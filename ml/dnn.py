"""
Module for creating and working with Deep Neural Networks.
No warranty!
"""
import itertools
import tempfile
import numpy as np
import tensorflow as tf
import utils.iterutils as iterutils
import utils.stats as stats
import utils.tfutils as tfutils


class DNN(object):
    """
    Implements (Deep) Neural Networks using TensorFlow.
    """
    def __init__(self, model_file=None, seed=None):
        """
        Creates a DNN object.
        If no arguments are supplied, an empty network graph is created.

        Args:
            model_file: The path to a model saved previously by a call to save. It is important that the meta file
                        exists as `model_file`.meta.
                        This argument should only be used when seed is None.
            seed: Sets the random seed of the tensorflow system.
                  This argument should only be used when model_file is None.
        """
        if seed and model_file:
            raise ValueError('Cannot give both a model_file and a seed')

        if model_file:
            self.load(model_file)
        else:
            self._config = tf.ConfigProto(allow_soft_placement=True)
            self._graph = tf.Graph()
            self._sess = tf.Session(config=self._config, graph=self._graph)

            if seed:
                with self._graph.as_default():
                    tf.set_random_seed(seed)
                    tf.add_to_collection('seed', seed)

            self._previous_layer = None

            self._neurons = []
            self._dropouts = []

            self.built = False

    def _init_weight(self, shape, mean=0.0, stddev=0.1):
        """
        Creates a tf Variable initialized with random variable in a normal distribution.

        Args:
            shape:  The shape of the Variable.
            mean:   The mean of the normal distribution.
            stddev: The standard deviation of the normal distribution.
        Returns:
            An initialized tf Variable with the specified shape
        """
        with self._graph.as_default():
            distribution = tf.random_normal(shape, mean=mean, stddev=stddev)
            return tf.Variable(distribution, name='variables')

    def _init_bias(self, shape, value=0.1):
        """
        Creates a tf Variable initialized to be good for use as a bias variable.

        Args:
            shape: The shape of the bias variable
            value: The value for the bias variable.
        Returns:
            An initialized bias variable.
        """
        initial = tf.constant(value, shape=shape)
        return tf.Variable(initial)

    def add_fc_layer(self, neurons, activation=tf.identity, dropout=None, bias=None):
        """
        Adds a fully connected layer to the neural network with a specified number of neurons.
        For the first layer, num_in must be specified, it is otherwise optional

        Args:
            neurons:  The number of neurons this layer should contain.
            activation: The activation function used by the layer.
            dropout:    The amount of dropout to be applied to the layer, specified as a value in the range [0,1].
                        If None, use no dropout.
            bias: A floating point number indicating what bias should be added to this layer. If None, use no bias.
        """

        if self.built:
            raise ValueError('Network is already built')

        with self._graph.as_default():
            if self._previous_layer is None:
                self._X = tf.placeholder(tf.float32, [None, neurons])
                layer = self._X
            else:
                layer = self._init_weight((self._neurons[-1], neurons))
                layer = tf.matmul(self._previous_layer, layer)

            if bias is not None:
                layer = tf.add(layer, self._init_bias((neurons,), bias))
            layer = activation(layer)

            if dropout is not None:
                drop = tf.placeholder(tf.float32)
                layer = tf.nn.dropout(layer, 1.-drop)
                tf.add_to_collection('dropout-{}'.format(len(self._dropouts)), drop)
                tf.add_to_collection('dropout-prob-{}'.format(len(self._dropouts)), dropout)
                self._dropouts.append((drop, dropout))

        self._previous_layer = layer
        self._neurons.append(neurons)

    def add_conv2d_layer(self, kernel_size, strides, padding, activation=tf.identity):
        pass

    def cost(self, data, batch_size=500):
        """
        Calculates the cost value of the data.

        Args:
            data: A tuple containing feats and labels.
            batch_size: The number of items to process at a time.
        """
        if not self.built:
            raise ValueError('Network not built yet')

        cost = 0.
        size = 0

        feed_dict = {}
        for (dropout) in self.dropouts:
            feed_dict[dropout] = 0

        tr_data = iterutils.grouper(data[0], batch_size)
        key_values = iterutils.grouper(data[1], batch_size)
        for (tr, key) in zip(tr_data, key_values):
            feed_dict[self._X] = tr
            feed_dict[self._Y] = key
            tm = self._sess.run(self.cost_op, feed_dict=feed_dict)
            cost += tm * len(tr)
            size += len(tr)
        cost /= size

        return cost

    def build(self,
              cost=tfutils.binary_cross_entropy,
              optimizer=tf.train.AdamOptimizer()):
        """
        Builds the neural network so that training and predicting can be made.
        The network must be at least two layers deep to be buildable.

        Args:
            cost: The cost function the be used for determining the loss value of the network.
                  Needs to be a function that takes two vectors as input and returns a value.
            optimizer: The optimizer to run over the network.
                       This needs to be an object that has a minimize function which takes a tensorflow
                       operation and updates the network based on it.
        """
        if self.built:
            raise ValueError('Network is already built')

        with self._graph.as_default():
            self._Y = tf.placeholder(tf.float32, [None, self._neurons[-1]])

            # For saving and restoring model later
            tf.add_to_collection('X', self._X)
            tf.add_to_collection('Y', self._Y)

            self.cost_op = tf.reduce_mean(cost(self._previous_layer, self._Y))
            self.train_op = optimizer.minimize(self.cost_op)
            self.predict_op = self._previous_layer

            # For saving and restoring model later
            tf.add_to_collection('cost', self.cost_op)
            tf.add_to_collection('train', self.train_op)
            tf.add_to_collection('predict', self.predict_op)

            init_op = tf.global_variables_initializer()
            self._sess.run(init_op)
            self.built = True

    def train(self, train, validation_data=None, epochs=100, batch_size=500,
              verbose=False, early_stopping=None):
        """
        Trains the network on the input data.
        Args:
            train: A tuple or list on the format [input, output] where input is
                   a list  of training data and output is a list of the correct
                   labels for the corresponding data.

                   Implementation specific detail:
                   generators can also be used as long as they are able to be
                   repeated. The data is accessed by calling (x0, x1) = train
                   each epoch, which can be utilized if neccessary.

            validation_data: Validation data, which is used to validate the
                             network each epoch.

            epochs: The number of epochs to train the network.

            batch_size: The number of items to use as network input at a time.
                        Most relevant when working on a GPU to determine the
                        number of data points to transfer to the GPU at a time.

            verbose: Determines if cost should be printed for each epoch.

            early_stopping: Determines if early stopping should be used.
                            If None, no early stopping will be used.
                            If it is an integer, training will stop after
                            `early_stopping` epochs of non-decreasing
                            validation cost.

                            Does not have an effect if there is no validation
                            set.
        """
        validation_string = ('\t  Validation acc(cutoff 0.4): {:.6f}\n'
                             '\t  Validation cost: {:.6f}')
        if not self.built:
            raise ValueError('Network not built yet')

        cost = 0

        tmp_storage_file = tempfile.NamedTemporaryFile()
        with self._graph.as_default():
            feed_dict = {}
            for dropout, p in self._dropouts:
                feed_dict[dropout] = p

            previous_val_cost = 999999999
            non_decreasing_epochs = 0
            for i in range(epochs):

                (train_data, train_keys) = train
                tr_data = iterutils.grouper(train_data, batch_size)
                key_values = iterutils.grouper(train_keys, batch_size)
                for (tr, key) in zip(tr_data, key_values):
                    feed_dict[self._X] = tr
                    feed_dict[self._Y] = key
                    self._sess.run(self.train_op, feed_dict=feed_dict)

                if verbose:
                    cost = self.cost(train, batch_size)
                    predictions = np.asarray(self.predict(train_data))
                    predictions[predictions > 0.4] = 1
                    predictions[predictions < 1.0] = 0
                    acc = stats.hamming_score(train_keys, predictions)
                    formatstring = '{} cost: {:.6f}  acc(cutoff 0.4): {:.6f}'
                    istr = str(i).zfill(len(str(epochs)))
                    print(formatstring.format(istr, cost, acc))

                if validation_data:
                    val_cost = self.cost(validation_data, batch_size)
                    if val_cost < previous_val_cost:
                        previous_val_cost = val_cost
                        non_decreasing_epochs = 0
                        if early_stopping:
                            self.save(tmp_storage_file.name)
                    else:
                        non_decreasing_epochs += 1
                    predictions = np.asarray(self.predict(validation_data[0]))
                    predictions[predictions > 0.4] = 1
                    predictions[predictions < 1.0] = 0
                    acc = stats.hamming_score(validation_data[1], predictions)
                    if verbose:
                        print(validation_string.format(acc, val_cost), end=' ')
                        if early_stopping:
                            epoch_left = early_stopping - non_decreasing_epochs
                            print('(Patience:', epoch_left, ')')
                        else:
                            print()

                    if not early_stopping:
                        continue
                    # Break out of training if we don't get improved
                    # performance on validation set
                    if (non_decreasing_epochs >= early_stopping):
                        self.load(tmp_storage_file.name)
                        break
        return cost

    def predict(self, data, batch_size=500):
        """
        Predicts on input data.
        Args:
            data:   A list of data on the same format that the network was
                    trained on.
            batch_size: How many items to predict on at a time.
        Returns:
            A list of predictions based on the internal classification model.
        """
        if not self.built:
            raise ValueError('Network not built yet')
        with self._graph.as_default():
            feed_dict = {}
            for dropout, _ in self._dropouts:
                feed_dict[dropout] = 0.

            predictions = []
            for tr in iterutils.grouper(data, batch_size):
                feed_dict[self._X] = tr
                tmp = self._sess.run(self.predict_op, feed_dict=feed_dict)
                predictions.extend(tmp)
            return predictions

    def save(self, path):
        """
        Saves the neural network model to file called `path`.
        Writes part of the model into a separate meta file, named `path`.meta.
        """
        with self._graph.as_default():
            saver = tf.train.Saver(sharded=False)
            saver.save(self._sess, path, write_meta_graph=True)

    def load(self, model_file):
        """
        Zeroes the current model and loads a previously trained one from file.
        It is important that both the file `model_file` and `model_file.meta`
        are available when loading the data.

        Args:
            model_file: The path from which to load the saved model.
                        Will also use the file model_file.meta
        """
        self._config = tf.ConfigProto(allow_soft_placement=True)
        self._graph = tf.Graph()
        self._sess = tf.Session(config=self._config, graph=self._graph)
        self._dropouts = []
        with self._graph.as_default():
            saver = tf.train.import_meta_graph(model_file + '.meta')
            saver.restore(self._sess, model_file)

            if len(tf.get_collection('seed')):
                tf.set_random_seed(tf.get_collection('seed')[0])

            self._X = tf.get_collection('X')[0]
            self._Y = tf.get_collection('Y')[0]
            self.predict_op = tf.get_collection('predict')[0]
            self.cost_op = tf.get_collection('cost')[0]
            self.train_op = tf.get_collection('train')[0]
            for i in itertools.count():
                dropout = tf.get_collection('dropout-%d' % i)
                dropout_prob = tf.get_collection('dropout-prob-%d' % i)
                if len(dropout):
                    self._dropouts.append((dropout[0], dropout_prob[0]))
                else:
                    break
        self.built = True
