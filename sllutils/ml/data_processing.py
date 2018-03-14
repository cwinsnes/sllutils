import collections
import numpy as np
import sllutils


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
        for class_ in self.classes:
            yield class_

    def __len__(self):
        return len(self.classes)

    __call__ = binarize


def cutoff_tuning(predictions, actual, cutoff_step=0.01, minimum_cutoff=0.07):
    """
    Calculates the cutoffs that should be used for each class in the predictions to get the best f1 score.
    The tuning is made by iterating over all possible predictions and as such is O(n^2).

    Useful when requiring your outputs to be binary but your ML method outputs real values.

    Note:
    In the parameters, m is the number of classes and n the number of items in your set.
    Args:
        predictions: A real valued (n x m) array, containing the predictions of the model.
        actual: A binary valued (n x m) array, containing the actual classes of the model.
        cutoff_step: How small increments should be made during tuning.
                     The smaller the increment, the more specific the tuning will be with an increased risk of
                     overfitting.
                     A tuning with a smaller cutoff step will also take longer to complete.
        minimum_cutoff: A convenience parameter, making sure that no cutoff value can be 0 as such values can make
                        later usage inaccurate of actual performance.
                        If no such minimum cutoff is wanted, minimum_cutoff should be set to 0.
    Returns:
        An array of size m with the cutoff value that yielded the best f1-score for each class.
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actual)
    best_results = [0.0] * len(actuals[0])
    best_cutoffs = [minimum_cutoff] * len(actuals[0])

    for cutoff in np.arange(minimum_cutoff, 1.0 + cutoff_step, cutoff_step):
        preds = []
        for prediction in predictions:
            prediction = prediction > cutoff
            preds.append(prediction)
        _, _, f1 = sllutils.utils.stats.precision_recall(preds, actuals, include_f1=True, mode='class')

        for i, f in enumerate(f1):
            if f > best_results[i]:
                best_results[i] = f
                best_cutoffs[i] = cutoff
    return best_cutoffs
