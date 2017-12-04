"""
Module for analyzing data and numbers.
"""
import numpy as np


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

# Alternative name to the jaccard index function
hamming_score = jaccard_index
