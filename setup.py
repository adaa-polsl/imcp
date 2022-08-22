from setuptools import setup

setup(name='mcp',
      version='0.1',
      description='Multiclass classification performance curve',
      author='Łukasz Wróbel, Bartosz Piguła',
      packages=['mcp'],
      install_requires=['numpy', 'matplotlib ~= 3.5.2'],
      classifiers=['Programming Language :: Python :: 3.8'],
      zip_safe=False)
