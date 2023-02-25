#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""enum_vals.py: Stores all enums required by Trainer-type Classes' 'params' dict"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import keras
from enum import auto, Enum


class Optimizers(Enum):
    """Defines optimizers, allows for shorthand references to function"""
    SGD      = keras.optimizers.SGD
    RMSPROP  = keras.optimizers.RMSprop
    ADAM     = keras.optimizers.Adam
    ADADELTA = keras.optimizers.Adadelta
    ADAGRAD  = keras.optimizers.Adagrad
    ADAMAX   = keras.optimizers.Adamax
    NADAM    = keras.optimizers.Nadam
    FTRL     = keras.optimizers.Ftrl

    @staticmethod
    def convert(optimizer_name: str):
        """Converts an input string to an optimizer, case insensitive"""
        return Optimizers[optimizer_name.upper()]


class TunerType(Enum):
    """Defines different tuning algorithms"""
    HYPERBAND = auto()
    BAYESIAN  = auto()


class Splits(Enum):
    """Defines the possible training splits"""
    TRAIN = "TRAIN"
    VAL   = "VAL"
    TEST  = "TEST"
    
    def __str__(self):
        """Allows for treating Enum val like any other const"""
        return str(self.value)
