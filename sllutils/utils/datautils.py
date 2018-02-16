def split(array, n):
    """
    Splits an array into two pieces.
    Args:
        array: The array to be split.
        n: An integer or float.
           If n is an integer, one of the arrays will contain n elements, the other will contain every other element.
           If n is a float, it needs to be between 0 and 1. The split will be such that one of the arrays will
           contain n*len(array) elements, the other will contain every other element.
    Returns:
        Two lists a, b.
        a is the list with the specified size.
        b is the list with all the other elements.
    """
    array = list(array)
    if isinstance(n, float):
        l = len(array)
        n = int(n * l)
    return array[:n], array[n:]
