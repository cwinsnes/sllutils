"""
Module for analyzing data and numbers.
"""
import numpy as np
import collections
import math


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
