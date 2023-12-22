
IMCP
========

Welcome to IMCP's documentation!

ROC curves are a well known tool for multiple classifier performance comparison. However, it does not work with multiclass datasets (more than two labels for the target variable). Moreover, the ROC curve is sensitive to imbalance of class distribution.

The package provides a tool - called Imbalanced Multiclass Classification Performance curve - that solves both weaknesses of ROC: application to multiclass and imbalanced datasets. 

With the IMCP curve the classification performance can be graphically shown for both multiclass and imbalanced datasets.

The package provides the methods for visualizing the IMCP curve and to provide the area under the IMCP curve.

Installation
============
IMCP can be installed from `PyPI <https://pypi.org/project/imcp/>`_::

   pip install imcp


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Tutorials <./rst/tutorials.rst>
   Code documentation <./rst/autodoc.rst>
   