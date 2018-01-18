"""
Module for util functions that are compaitible with Tensorflow.
"""
import os
import tensorflow as tf
from tensorflow.python.client import device_lib
import sys
from sllutils.utils.contextutils import redirect


def get_num_gpus():
    """
    Returns the number of available GPUs on this system.

    Note: Due to how tensorflow works, this function will do the following:
        1. Allocate a small amount of memory on each GPU
        2. Count number of GPUs
        3. Release GPU memory
        4. Return num gpus
    Running this function if any of these steps are unavailable will cause a crash.
    """
    with redirect(sys.stderr, os.devnull):
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.000001
        with tf.Session(config=config):
            gpu_devices = [device for device in device_lib.list_local_devices() if device.device_type == 'GPU']
    return len(gpu_devices)


def squared_error(prediction, target):
    """
    Calculates the squared error of the prediction against the target.

    (prediction - target)^2

    Args:
        prediction: The prediction vector.
        target: The target vector.
    Returns:
        The squared error of the prediction on the target, as per the above formula.
    """
    return tf.square(prediction - target)


def binary_cross_entropy(prediction, target):
    """
    Calculates the binary cross entropy value according to the below formula.
    The input vectors must be of the same length.

    let o=prediction, t=target
    -(t*log(o) + (1-t)*log(1-o))

    Args:
        prediction: The prediction vector. Values should be between 0 and 1.
        target: The target vector. This should be a binary vector using 0s and 1s.

    Returns:
        The binary cross entropy of the parameters, as defined by the above formula.

    Notes:
        Adds a small (1e-12) value to the logarithms to avoid log(0).
        As it is a loss function, values closer to 0 is better.
    """
    e = 1e-12
    op1 = tf.multiply(target, tf.log(prediction + e))
    op2 = tf.multiply(tf.subtract(1., target), tf.log(tf.subtract(1., prediction) + e))
    return tf.negative(tf.add(op1, op2))


def weighted_auxillary_cross_entropy(prediction, target, C=0.3):
    """
    Cross entropy version which includes another input as in the loss
    calculation.

    target should be a binary array containing two parts.
    The first half of the array should represent the true values, the "gold
    standard".
    The second half should represent the auxillary data.
    The array *has* to be evenly split by two.

    prediction should be a prediction array, at least as long as half the size
    of the target array.

    C can either be a floating point number or an array of floating point
    numbers. If it is an array, it has to be of equal length to half the size
    of the target array.

    The formula for calculation is roughly equivalent to:
        let t1 = target[0 : len(target)/2]
        let t2 = target[len(target)/2 : end(target)]
        let o = prediction[0 : len(target)/2]
        let BCE = binary_cross_entropy function
        loss = BCE(t1, o) + C * BCE(t2, o)


    Note:
        This cost function did not work out too well when we tried it out. Use
        at your own risk.
    """
    C = tf.constant(C, dtype=tf.float32)
    tensor_length = int(target.get_shape()[1])
    if tensor_length % 2:
        raise ValueError('Length of Tensor must be an even number')

    class_length = int(tensor_length/2)

    t1 = target[:, :class_length]
    t2 = target[:, class_length:]
    o = prediction[:, :class_length]

    bce1 = binary_cross_entropy(o, t1)
    bce2 = binary_cross_entropy(o, t2)

    gamer_influence = tf.multiply(C, bce2)

    return tf.add(bce1, gamer_influence)
