"""
Module for analyzing data and numbers.
"""
import numpy as np
import collections
import math


def precision_recall(prediction, actual, include_f1=False, mode='total'):
    """
    Calculates the precision and recall for a prediction on a dataset.
    Optionally calculates the f1 score as well.

    Args:
        prediction: A binary matrix representing the predictions on the dataset.
        actual: A binary matrix representing the actual true positives of the dataset.
        include_f1: Whether or not to include f1 in the return values.
        mode: One of 'total' or 'class'.
              In 'total' mode, the entire set is considered.
              In 'class' mode, the precision and recall is calculated for each class individually.
    Returns:
        A tuple containing (precision, recall).
        If include_f1 is True, the tuple contains (precision, recall, f1).
    """
    if mode == 'total':
        axis = None
    elif mode == 'class':
        axis = 0
    else:
        raise ValueError('The mode has to be either "total" or "class"')

    truepos = np.logical_and(prediction, actual)
    false = np.subtract(actual, prediction)
    falsepos = false < 0
    falseneg = false > 0

    truepos = np.sum(truepos, axis=axis)
    falsepos = np.sum(falsepos, axis=axis)
    falseneg = np.sum(falseneg, axis=axis)

    with np.errstate(divide='ignore', invalid='ignore'):
        precision = truepos/(truepos + falsepos)
        recall = truepos/(truepos + falseneg)
        if not np.isscalar(precision):
            precision[~np.isfinite(precision)] = 0
            recall[~np.isfinite(recall)] = 0

        if include_f1:
            f1 = 2*(precision*recall)/(precision+recall)

    if include_f1:
        return precision, recall, f1
    return precision, recall


def jaccard_index(y_true, y_predict):
    """
    Calculates the Jaccard index of the predictions on the true values.
    Also known as Jaccard similarity, Hamming score, or multi-label accuracy.

    Defined as:
    Let y_true=T, and y_predict=S.
    The Jaccard index  is calculated as
    |intersection(T,S)|/|union(T,S)|

    Args:
        y_true:   A list of binary vectors.
                  The list should consist of the target vectors.

        y_predict:   A list of binary vectors.
                     The list should consist of the prediction vectors.
    Returns:
        The Jaccard index (jaccard similarity) of the predictions on the true labels.
    """
    numerator = 0
    denominator = 0
    for (r, p) in zip(y_true, y_predict):
        if len(r) != len(p):
            raise ValueError('Array lengths do not agree')

        true = set(np.where(r)[0])
        pred = set(np.where(p)[0])

        intersection = true.intersection(pred)
        union = true.union(pred)
        numerator += len(intersection)
        denominator += len(union)
    return numerator/denominator


def shannon_entropy(sequence, log_base=2):
    """
    Calculates the shannon entropy over a sequence of values.

    Args:
        sequence: An iterable of values.
        log_base: What base to use for the logarithm.
                  Common values are: 2 (bits), e (nats), and 10 (bans).
    Returns:
        The shannon entropy over the sequence.
    """
    counter = collections.Counter(sequence)
    n = sum(counter.values())

    sequence_sum = 0
    for _, count in counter.items():
        sequence_sum += (count/n) * math.log(count/n, log_base)
    return -sequence_sum


# Alternative name to the jaccard index function
hamming_score = jaccard_index
