import setuptools
import os
import io

current_path = os.path.dirname(os.path.realpath(__file__))

with io.open(f"{current_path}/README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imcp",
    version="1.0.0",
    author="Łukasz Wróbel, Bartosz Piguła",
    description="Imbalanced Multiclass Classification Performance Curve",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/adaa-polsl/imcp",
    packages=["imcp"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved ::  BSD-3-Clause license",
        "Programming Language :: Python :: 3",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.6",
    install_requires=["numpy", "matplotlib"],
    test_suite="tests",
)
