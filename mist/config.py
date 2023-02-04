#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""config.py: Contains global configuration constants (e.g. path), ONLY ACCESS FROM __main__"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

from utils.enums import Optimizers, TunerType

from project_enums import TrainingModes, ModelType, TuneableParams


# Modify these
PATH = "C:/Users/hudso/Documents/Programming/Python/JH RI/MIST/" # Path to project
MODE = TrainingModes.GUI # "PLAIN", "DIST", "TUNER", "GUI", "DIST GUI", "TUNER GUI"

# Don't touch
EXPORT_DIR = "data" # Default is "data"
PREPROCESS_DATA = False # Default is False, set as True  to override existing data
LOAD_TUNER_PARAMS = False # Default is True, set as False to manually configure "params" var
PARAMS = {
    # General Hyperparameters
    "epochs":        50,
    "batch_size":    16,
    "learning_rate": 3.2e-4,
    "optimizer":     Optimizers.ADAM,
    # Training Constants
    "model_type":    ModelType.SDCC,
    "tuner_type":    TunerType.HYPERBAND, 
    "param_to_tune": TuneableParams.MODEL,
    # SDCC
    "filters":       6,
    "conv_layers":   5,
    "sdcc_blocks":   2,
    # BiLSTM
    "lstm_nodes":    200,
    "lstm_layers":   2,
    # Dense
    "dense_nodes":   320,
    "dense_layers":  1,
}
