# IMCP: Imbalanced Multiclass Classification Performance Curve

ROC curves are a well known tool for multiple classifier performance comparison. However, it does not work with multiclass datasets (more than two labels for the target variable). Moreover, the ROC curve is sensitive to imbalance of class distribution.

The package provides a tool - called Imbalanced Multiclass Classification Performance curve - that solves both weaknesses of ROC: application to multiclass and imbalanced datasets. 

With the IMCP curve the classification performance can be graphically shown for both multiclass and imbalanced datasets.

The package provides the methods for visualizing the IMCP curve and to provide the area under the IMCP curve.

## Installation

IMCP can be installed from [PyPI](https://pypi.org/project/imcp/)

```bash
pip install imcp
```

Or you can clone the repository and run:
```bash
pip install .
```

## Sample usage

```python
from imcp import plot_mcp_curve
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

clf = LogisticRegression(solver="liblinear").fit(X, y)
plot_mcp_curve(y, clf.predict_proba(X))
```

## Citation

The methodology is described in detail in:

[1] J. S. Aguilar-Ruiz and M. Michalak, “Diagnostic system assessment for imbalanced multiclass data” (in review, and draft available upon request). 

Also, the mathematical background of the multiclass classification performance can be found in:

[2] J. S. Aguilar-Ruiz and M. Michalak, "Multiclass Classification Performance Curve," in IEEE Access, vol. 10, pp. 68915-68921, 2022, doi: 10.1109/ACCESS.2022.3186444.

## Documentation

Full documentation is available [here](https://adaa-polsl.github.io/imcp/)
