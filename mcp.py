import numpy as np


def mcp_curve(y_true, y_score):
    """
    Parameters
    ----------
    y_true : array-like of shape (n_samples,). True labels.
    y_score : array-like of shape (n_samples, n_classes)
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class.
        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.
    """

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("'y_true' and 'y_score' have different number of samples")

    if not np.allclose(1, y_score.sum(axis=1)):
        raise ValueError(
            "Target scores need to be probabilities,"
            "i.e. they should sum up to 1.0 over classes"
        )

    y_true_size = np.max(y_true) + 1
    if y_true_size != y_score.shape[1]:
        raise ValueError(
            "Number of classes in 'y_true' not equal to the number of columns in 'y_score'"
        )

    # convert y_true to 0/1 array of scores (one hot encoding)
    y_true_score = np.eye(y_true_size)[y_true]

    #
    # mcp curve using Hellinger distance
    #

    # y-axis of the curve
    # sqrt for y_true_score is not required as it is 0/1 array
    curve_y = (y_true_score - np.sqrt(y_score)) ** 2
    curve_y = np.sum(curve_y, axis=1)
    curve_y = np.sqrt(curve_y) / np.sqrt(2)
    curve_y = 1 - curve_y
    curve_y = np.sort(curve_y)

    # x-axis of the curve
    n = len(curve_y)
    curve_x = np.arange(n)/(n - 1)

    return curve_x, curve_y
  
