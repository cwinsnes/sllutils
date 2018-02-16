import collections


class Binarizer(object):
    def __init__(self, classes):
        """
        Args:
            classes: A list of the classes that can be binarized.
        """
        self.classes = sorted(classes)
        self._index = dict(zip(self.classes, range(len(self.classes))))
        self._reverse_index = dict(zip(range(len(self.classes)), self.classes))

    def bin_label(self, item):
        """
        Binarize a single item.
        If the item is iterable and is not a string, the item will be binarized as a multi-label item.
        """
        bin_ = [0] * len(self.classes)
        if isinstance(item, collections.Iterable) and not isinstance(item, str):
            for c in item:
                bin_[self._index[c]] = 1
        else:
            bin_[self._index[item]] = 1
        return bin_

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
            bin_ = self.bin_label(item)
            binarized.append(bin_)
        return binarized

    def unbin_label(self, item):
        unbin = []
        for it in item:
            if it:
                unbin.append(self._reverse_index[it])
        return unbin

    def unbinarize(self, y):
        unbinarized = []
        for item in y:
            unbinarized.append(self.unbin_label(item))
        return unbinarized

    def __iter__(self):
        return self.classes

    def __len__(self):
        return len(self.classes)

    __call__ = binarize
