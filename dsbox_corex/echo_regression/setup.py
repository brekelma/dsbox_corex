#!/usr/bin/env python

from distutils.core import setup

setup(name='Echo Regression',
      version='0.1',
      description='Linear regression with fixed information capacity using echo noise.',
      author='Greg Ver Steeg',
      author_email='gversteeg@gmail.com',
      url='http://github.com/gregversteeg/echo_regression',
      packages=['numpy', 'scipy', 'scikit-learn', 'matplotlib'],
      )
