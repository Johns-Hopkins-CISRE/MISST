#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""datasets.py: Defines dataclass representations of datasets"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import keras
import numpy as np
from dataclasses import dataclass


@dataclass
class GeneratorDataset:
    """Represents generator-based datasets returned by _import_data"""
    train_gen: keras.utils.Sequence
    val_gen:   keras.utils.Sequence
    test_gen:  keras.utils.Sequence


@dataclass
class ArrayDataset:
    """Represents array-based datasets returned by _import_data"""
    x_train: np.ndarray | list 
    y_train: np.ndarray | list
    x_val:   np.ndarray | list
    y_val:   np.ndarray | list
    x_test:  np.ndarray | list 
    y_test:  np.ndarray | list
