import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union


def mcp_curve(y_true, y_score, labels: list = None, abs_tolerance=1e-8):
    """
    Calculate mcp curve using Hellinger distance.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). True labels.

    y_score : array-like of shape (n_samples, n_classes)
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

        If number of class labels in y_true differs from number of columns in
        y_score, a list with all labels must be given.

    labels : list with all class labels mapped to columns in y_score
        Must be given if any class is not represented in y_true, but y_score contains
        probabilities for this class.
        Number of labels must be equal to number of columns in y_score.
        All labels must be of the same dtype and share that dtype with y_true labels.

    abs_tolerance : absolute tolerance threshold for checking whether probabilities
        sum up to 1

    Returns
    ----------
    curve_x : numpy.array with x-coordinates of calculated curve

    curve_y : numpy.array with y-coordinates of calculated curve
    """

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("'y_true' and 'y_score' have different number of samples")

    if not np.allclose(1, y_score.sum(axis=1), rtol=0, atol=abs_tolerance):
        raise ValueError(
            "Target scores need to be probabilities,"
            "i.e. they should sum up to 1.0 over classes"
        )

    # check class labels and encode them as integers from 0 to y_true_size-1
    _, y_true_size, y_true_int_encoded = _map_class_labels(y_true, y_score, labels)

    # convert y_true to 0/1 array of scores (one hot encoding)
    y_true_score = np.eye(y_true_size)[y_true_int_encoded]

    # y-axis of the curve
    curve_y, _ = _get_y_values(y_true, y_true_score, y_score)

    # x-axis of the curve
    n = len(curve_y)
    curve_x = np.arange(n) / (n - 1)

    return (
        curve_x,
        curve_y,
    )


def imcp_curve(y_true, y_score, labels: list = None, abs_tolerance=1e-8):
    """
    Calculate imbalanced mcp curve using Hellinger distance. Unequal distribution
    of classes is taken into account.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). True labels.

    y_score : array-like of shape (n_samples, n_classes)
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

        If number of class labels in y_true differs from number of columns in
        y_score, a list with all labels must be given.

    labels : list with all class labels mapped to columns in y_score
        Must be given if any class is not represented in y_true, but y_score contains
        probabilities for this class.
        Number of labels must be equal to number of columns in y_score.
        All labels must be of the same dtype and share that dtype with y_true labels.

    abs_tolerance : absolute tolerance threshold for checking whether probabilities
        sum up to 1

    Returns
    ----------
    curve_x : numpy.array with x-coordinates of calculated curve

    curve_y : numpy.array with y-coordinates of calculated curve
    """

    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError("'y_true' and 'y_score' have different number of samples")

    if not np.allclose(1, y_score.sum(axis=1), rtol=0, atol=abs_tolerance):
        raise ValueError(
            "Target scores need to be probabilities,"
            "i.e. they should sum up to 1.0 over classes"
        )

    # check class labels and encode them as integers from 0 to y_true_size-1
    _, y_true_size, y_true_int_encoded = _map_class_labels(y_true, y_score, labels)

    # convert y_true to 0/1 array of scores (one hot encoding)
    y_true_score = np.eye(y_true_size)[y_true_int_encoded]

    # y-axis of the curve
    curve_y, sort_indices = _get_y_values(y_true, y_true_score, y_score)

    # x-axis of the curve
    class_widths = _get_class_widths(y_true_score, y_true_size)
    class_widths = class_widths[y_true_int_encoded]

    # sort in the same way as curve_y and get x-axis
    curve_x = class_widths[sort_indices]
    curve_x = np.cumsum(curve_x) - (curve_x / 2)

    # The first point (x = 0) and the last point (x = 1) retain the
    # first and last values of the Φ′ column (curve_y), i.e. these will be the
    # points (0, φ1 ) and (1, φn )
    curve_x = np.insert(curve_x, 0, 0)
    curve_x = np.append(curve_x, 1)
    curve_y = np.insert(curve_y, 0, curve_y[0])
    curve_y = np.append(curve_y, curve_y[-1])

    return curve_x, curve_y


def mcp_score(y_true, y_score, labels: list = None, abs_tolerance=1e-8):
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

        If number of class labels in y_true differs from number of columns in
        y_score, a list with all labels must be given.

    labels : list with all class labels mapped to columns in y_score
        Must be given if any class is not represented in y_true, but y_score contains
        probabilities for this class.
        Number of labels must be equal to number of columns in y_score.
        All labels must be of the same dtype and share that dtype with y_true labels.

    abs_tolerance : absolute tolerance threshold for checking whether probabilities
        sum up to 1

    Returns
    ----------
    area: Approximated area under curve
    """

    # calculate curve points
    curve_x, curve_y = mcp_curve(
        y_true, y_score, labels=labels, abs_tolerance=abs_tolerance
    )

    # integrate to get the area
    area = np.trapz(curve_y, x=curve_x)

    return area


def imcp_score(y_true, y_score, labels: list = None, abs_tolerance=1e-8):
    """
    Calculate area under imbalanced mcp curve with trapezoid rule.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). True labels.

    y_score : array-like of shape (n_samples, n_classes)
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class.

        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.

        If number of class labels in y_true differs from number of columns in
        y_score, a list with all labels must be given.

    labels : list with all class labels mapped to columns in y_score
        Must be given if any class is not represented in y_true, but y_score contains
        probabilities for this class.
        Number of labels pairs must be equal to number of columns in y_score.
        All labels must be of the same dtype and share that dtype with y_true labels.

    abs_tolerance : absolute tolerance threshold for checking whether probabilities
        sum up to 1

    Returns
    ----------
    area: Approximated area under curve
    """

    # calculate curve points
    curve_x, curve_y = imcp_curve(
        y_true, y_score, labels=labels, abs_tolerance=abs_tolerance
    )

    # integrate to get the area
    area = np.trapz(curve_y, x=curve_x)

    return area


def plot_mcp_curve(
    y_true,
    y_score,
    labels: list = None,
    abs_tolerance=1e-8,
    output_fig_path: str = None,
):
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

    labels : list with all class labels mapped to columns in y_score
        Must be given if any class is not represented in y_true, but y_score contains
        probabilities for this class.
        Number of labels must be equal to number of columns in y_score.
        All labels must be of the same dtype and share that dtype with y_true labels.

    abs_tolerance : absolute tolerance threshold for checking whether probabilities
        sum up to 1

    output_fig_path : if given, figure will be saved at this location. If no file extension is given,
        png will be used by default. Most backends support png, pdf, ps, eps and svg extensions.
    """
    x, y, plot_labels = [], [], []

    # check for data type
    if type(y_score) is dict:
        for key in y_score:
            area = np.around(
                mcp_score(
                    y_true,
                    y_score[key],
                    labels=labels,
                    abs_tolerance=abs_tolerance,
                ),
                decimals=4,
            )
            label = key + " (AUC={})".format(area)
            plot_labels.append(label)

            # get data for plotting
            x_array, y_array = mcp_curve(
                y_true,
                y_score[key],
                labels=labels,
                abs_tolerance=abs_tolerance,
            )
            x.append(x_array)
            y.append(y_array)
    else:
        x_array, y_array = mcp_curve(
            y_true, y_score, labels=labels, abs_tolerance=abs_tolerance
        )
        x.append(x_array)
        y.append(y_array)
        area = np.around(
            mcp_score(y_true, y_score, labels=labels, abs_tolerance=abs_tolerance),
            decimals=4,
        )
        label = "clf1 (AUC={})".format(area)
        plot_labels.append(label)

    fig_title = "MCP curves" if len(x) > 1 else "MCP curve"
    plot_curve(x, y, label=plot_labels, output_fig_path=output_fig_path, fig_title=fig_title)


def plot_imcp_curve(
    y_true,
    y_score,
    labels: list = None,
    abs_tolerance=1e-8,
    output_fig_path: str = None,
):
    """
    Plot imbalanced mcp curve based on given probabilities and labels. If more than one algorithm scores given,
    plot all curves.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,). True labels.

    y_score : array-like of shape (n_samples, n_classes) or dictionary with algorithm's label key and array-like value.
        Target scores corresponding to probability estimates of a sample
        belonging to a particular class.
        The order of the class scores must correspond to the numerical or
        lexicographical order of the labels in y_true.
        If dictionary passed, a curve is plot for each key-value pair existing in dict.

    labels : list with all class labels mapped to columns in y_score
        Must be given if any class is not represented in y_true, but y_score contains
        probabilities for this class.
        Number of labels must be equal to number of columns in y_score.
        All labels must be of the same dtype and share that dtype with y_true labels.

    abs_tolerance : absolute tolerance threshold for checking whether probabilities
        sum up to 1

    output_fig_path : if given, figure will be saved at this location. If no file extension is given,
        png will be used by default. Most backends support png, pdf, ps, eps and svg extensions.
    """
    x, y, plot_labels = [], [], []

    # check for data type
    if type(y_score) is dict:
        for key in y_score:
            area = np.around(
                imcp_score(
                    y_true,
                    y_score[key],
                    labels=labels,
                    abs_tolerance=abs_tolerance,
                ),
                decimals=4,
            )
            label = key + " (AUC={})".format(area)
            plot_labels.append(label)

            # get data for plotting
            x_array, y_array = imcp_curve(
                y_true,
                y_score[key],
                labels=labels,
                abs_tolerance=abs_tolerance,
            )
            x.append(x_array)
            y.append(y_array)
    else:
        x_array, y_array = imcp_curve(
            y_true, y_score, labels=labels, abs_tolerance=abs_tolerance
        )
        x.append(x_array)
        y.append(y_array)
        area = np.around(
            imcp_score(y_true, y_score, labels=labels, abs_tolerance=abs_tolerance),
            decimals=4,
        )
        label = "clf1 (AUC={})".format(area)
        plot_labels.append(label)


    fig_title = "IMCP curves" if len(x) > 1 else "IMCP curve"
    plot_curve(x, y, label=plot_labels, output_fig_path=output_fig_path, fig_title=fig_title)


def plot_curve(
    x,
    y,
    label: Union[str, List[str]] = None,
    output_fig_path: str = None,
    fig_title = "(I)MCP curve(s)"
):
    """
    Plot curves described with given x and y coordinates.
    To plot multiple curves, pass x and y as 2D arrays. Each row will be plotted
    as a separate curve.

    Parameters
    ----------
    x : array-like or array of arrays.
    y : array-like or array of arrays. Must be of the same shape as x.
    label : single label or list of labels, which will be displayed on the plot as legend
    output_fig_path : if given, figure will be saved at this location. If no file extension is given,
        png will be used by default. Most backends support png, pdf, ps, eps and svg extensions.
    """
    ## parsing inputs
    # dimensions of x and y
    x_dim = _get_dimensions(x)
    y_dim = _get_dimensions(y)
    if x_dim != y_dim:
        raise ValueError("x and y have different dimensions")

    if x_dim == 1:
        curve_x = [x]
        curve_y = [y]
    elif x_dim == 2:
        if len(x) != len(y):
            raise ValueError("x and y have inconsistent lengths")

        for idx in range(len(x)):
            if len(x[idx]) != len(y[idx]):
                raise ValueError(
                    f"x and y at index {idx} have different number of samples"
                )

        curve_x = x
        curve_y = y
    else:
        raise ValueError("x and y should be 1d or 2d arrays")

    # user-defined labels
    if label is not None:
        if type(label) is str:
            if x_dim > 1:
                raise ValueError("Single label given for 2D x and y")
            else:
                labels = [label]
        elif type(label) is list:
            if len(x) != len(label):
                raise ValueError("Number of labels is different than number of curves")
            if any(type(val) is not str for val in label):
                raise ValueError("Labels contain non-string objects")
            labels = label
        else:
            raise ValueError("Unsupported type for label")

    # output path
    if output_fig_path is not None:
        if type(output_fig_path) is not str:
            raise ValueError("Given path is not a string")

    ## plot curves
    fig, ax = plt.subplots(1, figsize=[9, 7])
    for idx in range(len(curve_x)):
        current_label = labels[idx] if label is not None else None
        ax.plot(
            curve_x[idx],
            curve_y[idx],
            label=current_label,
            linestyle="solid",
            color=_get_color(idx),
        )
    fig.suptitle(fig_title)

    major_ticks = np.arange(0, 1.05, 0.2)
    minor_ticks = np.arange(0, 1.05, 0.1)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid()
    ax.grid(which="minor", alpha=0.6)

    # setting style cosmetics
    ticks = np.arange(0, 1.1, step=0.1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.grid(
        which="both",
        axis="both",
        color=(0.501960784313725, 0.501960784313725, 0.501960784313725),
        linestyle="dotted",
        linewidth=0.5,
    )

    color = (0.5, 0.5, 0.5)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_linewidth(0.5)
    ax.spines.bottom.set_color(color)
    ax.spines.left.set_linewidth(0.5)
    ax.spines.left.set_color(color)

    ax.set_xlim([-0.005, 1.001])
    ax.set_ylim([-0.005, 1.001])

    ax.xaxis.label.set_color(color)
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis="x", colors=color)
    ax.tick_params(axis="y", colors=color)

    ax.set_aspect("equal")

    if label is not None:
        plt.legend()

    if output_fig_path is not None:
        plt.savefig(output_fig_path, bbox_inches="tight")
    else:
        plt.show()


def _get_y_values(y_true, y_true_score, y_score):
    """y-axis of the mcp curve using Hellinger distance"""
    # sqrt for y_true_score is not required as it is 0/1 array (one hot encoded)
    curve_y = (y_true_score - np.sqrt(y_score)) ** 2
    curve_y = np.sum(curve_y, axis=1)
    curve_y = np.sqrt(curve_y) / np.sqrt(2)
    curve_y = 1 - curve_y

    # indices can be also used to sort curve_x (imcp)
    sort_indices = np.lexsort((y_true, curve_y))
    curve_y = np.array(curve_y)[sort_indices]

    return curve_y, sort_indices


def _map_class_labels(y_true, y_score, labels):
    """Check the number of classes in given arrays and encode class labels as integers from 0 to n-1"""
    unique_classes, y_true_int_encoded = np.unique(y_true, return_inverse=True)
    y_true_size = len(unique_classes)
    class_mapper = {label: index for index, label in enumerate(unique_classes)}

    if y_true_size != y_score.shape[1]:
        if labels is None:
            raise ValueError("Class labels not given!")
        else:
            if len(labels) != y_score.shape[1]:
                raise ValueError(
                    "Number of class labels not equal to the number of columns in 'y_score'"
                )

            if np.array(labels).dtype != y_true.dtype:
                raise TypeError(
                    "Given labels and values in y_true are of different types"
                )

            if not set(unique_classes).issubset(set(labels)):
                raise KeyError(
                    "Class labels from y_true are not a subset of given list of labels. Check if values and types of given labels and y_true match."
                )

            unique_classes = np.sort(labels)
            y_true_size = len(unique_classes)
            class_mapper = {label: index for index, label in enumerate(unique_classes)}
            y_true_int_encoded = np.vectorize(class_mapper.get)(y_true)

    return class_mapper, y_true_size, y_true_int_encoded


def _get_class_widths(y_true_score, y_true_size):
    """Width of example ei depending on its class is defined as wi = 1/(no_classes * ei_class_count)"""
    class_widths = np.sum(y_true_score, axis=0)
    class_widths = 1 / (y_true_size * class_widths)

    return class_widths


def _get_color(idx: int = 0) -> tuple:
    """Get color from predefined palette"""
    palette = [
        (135 / 255, 181 / 255, 222 / 255),
        (170 / 255, 122 / 255, 122 / 255),
        (240 / 255, 128 / 255, 128 / 255),  # lightcoral
        (34 / 255, 139 / 255, 34 / 255),  # forestgreen
        (238 / 255, 130 / 255, 238 / 255),  # violet
        (220 / 255, 20 / 255, 60 / 255),  # crimson
        (139 / 255, 69 / 255, 19 / 255),  # saddlebrown
        (255 / 255, 140 / 255, 0 / 255),  # darkorange
        (65 / 255, 105 / 255, 225 / 255),  # royalblue
        (64 / 255, 224 / 255, 208 / 255),  # turquoise
        (95 / 255, 158 / 255, 160 / 255),  # cadetblue
        (30 / 255, 144 / 255, 255 / 255),  # dodgerblue
        (138 / 255, 43 / 255, 226 / 255),  # blueviolet
        (124 / 255, 252 / 255, 0 / 255),  # lawngreen
        (255 / 255, 20 / 255, 147 / 255),  # deeppink
        (0 / 255, 255 / 255, 0 / 255),  # lime
        (189 / 255, 183 / 255, 107 / 255),  # darkkhaki
        (0, 0, 0),
    ]

    if idx >= 0 and idx < len(palette):
        return palette[idx]
    else:
        color_value = abs(idx) % 256
        return (color_value, color_value, color_value)


def _get_dimensions(lst):
    if isinstance(lst, list) or isinstance(lst, np.ndarray):
        if isinstance(lst, list):
            return 1 + max(_get_dimensions(sublist) for sublist in lst)
        else:
            return len(lst.shape)
    else:
        return 0
