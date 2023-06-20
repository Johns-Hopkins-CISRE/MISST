#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""enums.py: Stores project-specific enums used by ModelTrainer's 'params' dict"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

from enum import auto, Enum


class TrainingModes(Enum):
    """Defines all possible modes of training"""
    # Pure modes
    PLAIN = auto()
    DIST = auto()
    GUI = auto()
    TUNER = auto()
    # Composite modes
    DIST_GUI = auto()
    TUNER_GUI = auto()


class ModelType(Enum):
    """Defines all model types"""
    BOTTLENECK = auto()
    SDCC = auto()

    @staticmethod
    def convert(model_name: str):
        """Converts an input string to a model type, case insensitive"""
        return ModelType[model_name.upper()]


class TuneableParams(Enum):
    """Defines the parameters that the Tuner adjusts"""
    MODEL = auto()
    LR = auto()
