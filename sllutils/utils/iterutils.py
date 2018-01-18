"""
Module for general iteration helpers.
"""
import itertools


def grouper(iterable, n):
    """
    Iterates over an iterable in chunks of size n.

    Args:
        iterable (iterator):   The iterable to iterate over.
        n (int):  The chunk size.
    Returns:
        A generator which will yield the entire iterable in chunks of size n.

    Notes:
    Copied from https://stackoverflow.com/questions/8991506/iterate-an-iterator-by-chunks-of-n-in-python
    Thanks to Sven Marnach for the code snippet.
    """
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def flatten(iterable, n=1):
    """
    Flattens an iterable n times.

    Args:
        iterable: The list to flatten.
        n: The number of times the list should be flattened.

    Returns:
        A generator yielding the flattened list.
    """
    for _ in range(n):
        iterable = itertools.chain.from_iterable(iterable)
    return iterable
