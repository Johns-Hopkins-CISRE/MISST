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
MODE = TrainingModes.TUNER # "PLAIN", "DIST", "TUNER", "GUI", "DIST GUI", "TUNER GUI"

# Don't touch
EXPORT_DIR = "data" # Default is "data"
PREPROCESS_DATA = False # Default is False, set as True  to override existing data
LOAD_TUNER_PARAMS = False # Default is True, set as False to manually configure "params" var

# General Hyperparameters
MODEL_PARAMS: ModelParams = {
    "epochs":        50,
    "batch_size":    16,
    "learning_rate": 3.2e-4,
    "optimizer":     Optimizers.ADAM,
    "model_type":    ModelType.SDCC,

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
    }
}

# "TunerTrainer"-specific Parameters
TUNER_PARAMS: TunerParams = { 
    "tuner_type":    TunerType.HYPERBAND, 
    "param_to_tune": TuneableParams.MODEL,
    "goal":          "val_sparse_categorical_accuracy",
    "dir_name":      "tuner_results",

    "tuner_configs": {
        TunerType.HYPERBAND: {
            "factor": 3
        },
        TunerType.BAYESIAN: {
            "num_trials": 20
        }
    }
}