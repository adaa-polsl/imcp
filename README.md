# **Imbalanced Multiclass Classification Performance (IMCP) Curve**
ROC curves are a well known tool for multiple classifier performance comparison. However, it does not work with multiclass datasets (more than two labels for the target variable). Moreover, the ROC curve is sensitive to imbalance of class distribution.

The package provides a tool - called Imbalanced Multiclass Classification Performance curve - that solves both weaknesses of ROC: application to multiclass and imbalanced datasets. 

With the IMCP curve the classification performance can be graphically shown for both multiclass and imbalanced datasets.

The package provides the methods for visualizing the IMCP curve and to provide the area under the IMCP curve.

The methodology is described in detail in:

[1] J. S. Aguilar-Ruiz and M. Michalak, “Diagnostic system assessment for imbalanced multiclass data” (in review, and draft available upon request). 

Also, the mathematical background of the multiclass classification performance can be found in:

[2] J. S. Aguilar-Ruiz and M. Michalak, "Multiclass Classification Performance Curve," in IEEE Access, vol. 10, pp. 68915-68921, 2022, doi: 10.1109/ACCESS.2022.3186444.

We’d like to thank our colleagues - Łukasz Wróbel and Bartosz Piguła - for their participation in the Python package development.



