#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

version = "0.1.1"
setup(name='hierarchical_grid_search_cv',
      version=version,
      description='Python implementations of metric learning algorithms',
      author=['Ondrej Svec'],
      author_email='ond.svec@gmail.com',
      url='http://github.com/svecon/HierarchicalGridSearchCV',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering'
      ],
      packages=['hierarchical_grid_search_cv'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'six'
      ],
      extras_require=dict(
      ),
      test_suite='test',
      keywords=[
          'Metric Learning',
          'Large Margin Nearest Neighbor',
          'Information Theoretic Metric Learning',
          'Sparse Determinant Metric Learning',
          'Least Squares Metric Learning',
          'Neighborhood Components Analysis'
      ])