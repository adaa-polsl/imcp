import numpy as np
import matplotlib.pyplot as plt


def mcp_curve(y_true, y_score, abs_tolerance=1e-8):
    """
    Parameters
    ----------
    y_true : array-like of shape (n_samples,). True labels.
    y_score : array-like of shape (n_samples, n_classes)
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class.
        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.
    abs_tolerance : absolute tolerance threshold for checking whether probabilities 
        sum up to 1
    """

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("'y_true' and 'y_score' have different number of samples")

    if not np.allclose(1, y_score.sum(axis=1), rtol=0, atol=abs_tolerance):
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


def mcp_score(y_true, y_score, abs_tolerance=1e-8):
    """
    Calculate area under mcp curve with trapezoid rule.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,). True labels.
    y_score : array-like of shape (n_samples, n_classes)
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class.
        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.
    abs_tolerance : absolute tolerance threshold for checking whether probabilities 
        sum up to 1
    Returns
    ----------
    area: Approximated area under curve
    """
    
    # calculate curve points
    curve_x, curve_y = mcp_curve(y_true, y_score, abs_tolerance=abs_tolerance)
    
    # integrate to get the area
    area = np.trapz(curve_y, x=curve_x)
    
    return area
    

def plot_mcp_curve(y_true, y_score, abs_tolerance=1e-8):
    """
    Plot mcp curve based on given probabilities and labels. If more than one algorithm scores given, plot all curves.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,). True labels.
    y_score : array-like of shape (n_samples, n_classes) or dictionary with algorithm's label key and array-like value.
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class.
        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.
        If dictionary passed, a curve is plot for each key-value pair existing in dict.
    abs_tolerance : absolute tolerance threshold for checking whether probabilities 
        sum up to 1
    """
    x, y, labels = [], [], []
    
    # check for data type
    if type(y_score) is dict:
        for key in y_score:
            area = np.around(mcp_score(y_true, y_score[key], abs_tolerance=abs_tolerance),
                    decimals=4)
            label = key + " (AUC={})".format(area)
            labels.append(label)
            
            # get data for plotting
            x_array, y_array = mcp_curve(y_true, y_score[key], abs_tolerance=abs_tolerance)
            x.append(x_array)
            y.append(y_array)
    else:
        x_array, y_array = mcp_curve(y_true, y_score, abs_tolerance=abs_tolerance)
        x.append(x_array)
        y.append(y_array)
        area = np.around(mcp_score(y_true, y_score, abs_tolerance=abs_tolerance),
                decimals=4)
        label = 'clf1 (AUC={})'.format(area)
        labels.append(label)
        
    # plot curves
    for idx in range(len(x)):
        plt.plot(x[idx], y[idx], label=labels[idx])
    plt.title('Mcp curve(s)')
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.show()
        
