#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""__main__.py: Preprocesses and trains the model as per the config.py file"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import pickle
import os

from config import *
from project_enums import TrainingModes
from model_trainer import DistributedGUI, ModelTrainer
from preprocessor import PreProcessor


# Preprocesses data
if PREPROCESS_DATA:
    preproc = PreProcessor(PATH)
    preproc.import_and_preprocess()
    preproc.regroup()
    preproc.group_shuffle()
    preproc.split_dataset()
    preproc.save_len()

# Defines training parameters
model_params = MODEL_PARAMS
if LOAD_TUNER_PARAMS:
    os.chdir(f"{PATH}data/")
    filename = f"hps_{TUNER_FILE_TO_LOAD['tuner_type'].name}_{model_params['model_type'].name}_{TUNER_FILE_TO_LOAD['tuned_params'].name}.pkl"
    with open(filename, "rb") as f:
        best_hps = pickle.load(f)
    match TUNER_FILE_TO_LOAD["tuned_params"]:
        case TuneableParams.MODEL:
            model_params["archi_params"][model_params["model_type"]].update(best_hps)
        case TuneableParams.LR:
            model_params.update(best_hps)

# Runs training according to declared training method
match MODE:
    case TrainingModes.PLAIN:
        trainer = ModelTrainer(PATH, EXPORT_DIR, model_params)
        trainer.basic_train()
    case TrainingModes.DIST:
        trainer = ModelTrainer(PATH, EXPORT_DIR, model_params)
        trainer.dist_train()
    case TrainingModes.TUNER:
        trainer = ModelTrainer(PATH, EXPORT_DIR, model_params, TUNER_PARAMS)
        trainer.tuner_train()
    case TrainingModes.GUI | TrainingModes.DIST_GUI | TrainingModes.TUNER_GUI:
        DistributedGUI(PATH, EXPORT_DIR, MODE)
    case other:
        raise ValueError(f"Variable \"MODE\" is invalid, got val \"{MODE}\"")
