#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""config.py: Contains global configuration constants (e.g. path), ONLY ACCESS FROM __main__"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

from utils.enum_vals import Optimizers, TunerType
from utils.req_params import ModelParams, TunerParams
from project_enums import TrainingModes, ModelType, TuneableParams


# Modify these
PATH = "C:/Users/hudso/Documents/Programming/Python/JH RI/MIST/" # Path to project
MODE = TrainingModes.PLAIN # "PLAIN", "DIST", "TUNER", "GUI", "DIST GUI", "TUNER GUI"

# Don't touch
EXPORT_DIR = "data" # Default is "data"
PREPROCESS_DATA = False # Default is False, set as True  to override existing data
LOAD_TUNER_PARAMS = False # Default is True, set as False to manually configure "params" var
TUNER_FILE_TO_LOAD = { # Irrelevant if LOAD_TUNER_PARAMS is False
    "tuner_type":   TunerType.HYPERBAND,
    "tuned_params": TuneableParams.MODEL
}

# General Hyperparameters
MODEL_PARAMS: ModelParams = {
    "epochs":        200,
    "batch_size":    16,
    "learning_rate": 2e-3, #3.2e-4
    "decay_steps":   40,
    "optimizer":     Optimizers.ADAM,
    "model_type":    ModelType.BOTTLENECK,

    "archi_params": {
        ModelType.SDCC: {
            "filters":       6,
            "conv_layers":   5,
            "sdcc_blocks":   2,

            "lstm_nodes":    200,
            "lstm_layers":   2,

            "dense_nodes":   320,
            "dense_layers":  1,
        },
        ModelType.BOTTLENECK: {
            "init_kernel":    16,

            "cnn_blocks":     4,
            "bn_blocks":      3, # Abbrev. for "Bottleneck"
            "conv_pattern":   [1,3], # Mirrored: 1,3,4 -> 1,3,4,3,1
            
            "filter_mult":    16,  # Filter multiplier
            "scaling_factor": 4  # Factor by which init val is multiplied and later compensation is applied
        }
    }
}

# "TunerTrainer"-specific Parameters
TUNER_PARAMS: TunerParams = { 
    "tuner_type":     TunerType.HYPERBAND, 
    "params_to_tune": TuneableParams.MODEL,
    "goal":           "val_sparse_categorical_accuracy",
    "dir_name":       "tuner_results",

    "tuner_configs": {
        TunerType.HYPERBAND: {
            "max_epochs": 1,
            "factor": 3
        },
        TunerType.BAYESIAN: {
            "num_trials": 20
        }
    }
}