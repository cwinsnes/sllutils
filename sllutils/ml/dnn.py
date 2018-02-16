"""
Module for creating and working with Deep Neural Networks.
No warranty!
"""
import tensorflow as tf
import sllutils.utils.tfutils as tfutils
import sllutils.utils.contextutils as contextutils
import sys
import shutil
import os
with contextutils.redirect(sys.stderr, os.devnull):
    import keras


class MultiGpuCheckpointCallback(keras.callbacks.Callback):
    def __init__(self, path, model):
        self.path = path
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        self.model_to_save.save(self.path + '_at_epoch_%d.h5' % epoch)


class DNN(object):
    # Yes, this is basically a wrapper around a wrapper of tensorflow...
    """
    Implements (Deep) Neural Networks using Keras and Tensorflow.
    """
    def __init__(self, model_on_cpu=True):
        """
        Args:
            model_on_cpu: A boolean indicating whether the model should be created on the CPU or if the backend
                          should decide where to put it.

                          For single-GPU training this must be `False` as the model will otherwise not utilize the
                          GPU at all.

                          For multi-GPU training, `model_on_cpu=True` is recommended as the variable synchronization
                          might end up on a GPU otherwise.

                          For CPU-training, it should not matter at all as the model should be placed on the CPU
                          anyways.
        """
        self.model = keras.models.Sequential()
        self._input_shape = None
        self._built = False
        self._first_layer = True
        self._model_on_cpu = model_on_cpu

    def _add_layer(self, layer, *args, **kwargs):
        if self._built:
            raise ValueError('Model is already built')
        if self._first_layer:
            kwargs['input_shape'] = self._input_shape
        else:
            raise ValueError('No input shape is specified')

        if self._model_on_cpu:
            with tf.device('/cpu:0'):
                self.model.add(layer(*args, **kwargs))
        else:
            self.model.add(layer(*args, **kwargs))

    def set_input_shape(self, shape):
        """
        Sets the input layer of the network.

        Args:
            shape: a tuple determining the size of the input layer.
                   For variable number of inputs, this list should begin with a 'None'.
                   For example, MNIST could use the shapes (784) or (28, 28)
        """
        if self._built:
            raise ValueError("Model is already built")
        if self._input_shape:
            raise ValueError("Input shape already set")
        self._input_shape = shape

    def flatten(self):
        """
        Flattens the previous layer to a 1-dimensional tensor.
        """
        self._add_layer(keras.layers.Flatten)

    def reshape(self, shape):
        """
        Reshapes the previous layer to the desired shape.
        Args:
            shape: The desired new shape for the tensor.
        """
        self._add_layer(keras.layers.Reshape, shape)

    def conv2D(self, filters, kernel_size, strides=(1, 1), dilation_rate=(1, 1), activation=None):
        """
        Adds a convolutional2D layer to the neural network.
        Args:
            filters: The convolutional filters to be used.
                     If this is a single number, use the same filter in all dimensions.
                     Otherwise this should be a tuple of two integers, one for each dimension.

            kernel_size: The kernel size to be used.
                         If this is a single number, use the same kernel size in all dimensions.
                         Otherwise this should be a tuple of two integers, one for each dimension.
            activation: The name of the activation function to use.
                        If None, linear activation is used (e.g. A(x) = x).
        """
        self._add_layer(keras.layers.Conv2D, filters, kernel_size, strides=strides,
                        dilation_rate=dilation_rate, activation=activation)

    def activation(self, activation):
        """
        Adds an activation to the previous layer.
        Args:
            activation: The name of the activation function to use.
        """
        self._add_layer(keras.layers.Activation, activation)

    def batchnormalization(self):
        self._add_layer(keras.layers.BatchNormalization)

    def maxpool2D(self, pool_size=(2, 2), strides=None):
        self._add_layer(keras.layers.MaxPooling2D, pool_size, strides)

    def dropout(self, drop_rate):
        self._add_layer(keras.layers.Dropout, drop_rate)

    def dense(self, neurons, activation=None):
        self._add_layer(keras.layers.Dense, neurons, activation=activation)

    def build(self, loss, optimizer, num_gpus=None):
        """
        Args:
            num_gpus: The number of gpus to use for this model.
                      If None, use the maximum number of available GPUs.
        """
        self._base_model = self.model
        if num_gpus is None:
            num_gpus = tfutils.get_num_gpus()
        if num_gpus >= 2:
            self.model = keras.utils.multi_gpu_model(self.model, num_gpus)
        self.model.compile(optimizer=optimizer, loss=loss)
        self._built = True

    def train(self, x, y, batch_size=None, epochs=1, validationx=None, validationy=None, verbose=False):
        for epoch in range(epochs):
            hist = self.model.fit(x, y, batch_size, verbose=0)
            if verbose:
                print('{}: {}'.format(epoch, hist.history['loss'][-1]))

    def train_generator(self, generator, batches_per_epoch, epochs=1, validation_generator=None,
                        validation_batches_per_epochs=None, verbose=False, early_stopping=None,
                        checkpointing_folder=None):
        callbacks = []
        if early_stopping:
            early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                                patience=early_stopping, verbose=1, mode='auto')
            callbacks.append(early_stop_callback)

        if checkpointing_folder:
            try:
                os.makedirs(checkpointing_folder)
            except OSError:
                pass
            checkpointing = os.path.join(checkpointing_folder, 'weights')
            checkpoint_callback = MultiGpuCheckpointCallback(checkpointing, self._base_model)
            callbacks.append(checkpoint_callback)

        self.model.fit_generator(generator,
                                 batches_per_epoch,
                                 validation_data=validation_generator,
                                 validation_steps=validation_batches_per_epochs,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 verbose=1)

    def predict(self, x, batch_size=None):
        return self.model.predict(x, batch_size)

    def save(self, path):
        try:
            os.makedirs(path)
        except OSError:
            pass

        modelpath = os.path.join(path, 'model.json')
        weightpath = os.path.join(path, 'weights.h5')

        if os.path.exists(modelpath):
            shutil.move(modelpath, modelpath + '_bak')
        if os.path.exists(weightpath):
            shutil.move(weightpath, weightpath + '_bak')

        model_json = self._base_model.to_json()
        open(modelpath, 'w').write(model_json)
        self._base_model.save_weights(weightpath)

    def load_model(self, path):
        modelpath = os.path.join(path, 'model.json')
        weightpath = os.path.join(path, 'weights.h5')

        model_json = open(modelpath, 'r').read()
        model = keras.models.model_from_json(model_json)

        model.load_weights(weightpath)
        self.model = model
        self._built = True

    @classmethod
    def load(cls, path):
        with contextutils.redirect(sys.stderr, os.devnull):
            instance = cls()
            instance.load_model(path)
        return instance
