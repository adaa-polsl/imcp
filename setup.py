from setuptools import setup

setup(
    name="imcp",
    version="0.21",
    description="Imbalanced multiclass classification performance curve",
    author="Łukasz Wróbel, Bartosz Piguła",
    packages=["imcp"],
    install_requires=["numpy", "matplotlib ~= 3.5.2", "pandas"],
    classifiers=["Programming Language :: Python :: 3.8"],
    zip_safe=False,
)
