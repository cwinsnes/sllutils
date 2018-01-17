import collections


class Binarizer(object):
    def __init__(self, classes):
        """
        Args:
            classes: A list of the classes that can be binarized.
        """
        self._classes = classes
        self._index = dict(zip(self._classes, range(len(self._classes))))

    def binarize(self, y):
        """
        Args:
            y: A list of of labels to be binarized.
               Items in `y` that are iterable (except strings) will be binarized as a multi-label item,
               all other items will be binarized as a single-label item.
        Returns:
            A list of binarized label lists.
        """
        binarized = []
        for item in y:
            bin_ = [0] * len(self._classes)
            if isinstance(item, collections.Iterable) and not isinstance(item, str):
                for c in item:
                    bin_[self._index[c]] = 1
            else:
                bin_[self._index[item]] = 1
            binarized.append(bin_)
        return binarized

    __call__ = binarize
