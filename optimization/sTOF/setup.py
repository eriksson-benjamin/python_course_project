#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 13:12:43 2021

@author: beriksso
"""

from distutils.core import setup
from Cython.Build import cythonize

setup(
      ext_modules = cythonize("CyTOF.pyx")
      )