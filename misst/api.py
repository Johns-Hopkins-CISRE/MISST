#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""api.py: Defines high-level methods for running MISST"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import pickle
import os

from misst.trainer.utils.error_handler import short_err

from misst.trainer.model_trainer import DistributedGUI, ModelTrainer
from misst.trainer.preprocessor import PreProcessor

from misst.predictor import predictor


def __validate_yaml_values(config: dict):
    """
    Ensures that all the YAML config values are logically consistent.
    Note that this does not check for valid key-value pairs; it is
    assumed that all requisite key-value pairs exist.
    """
    # Quickly define the temporary model parameter variable
    model_params = config["model_params"]

    # Check for model-type mismatch
    if model_params["model_type"] not in model_params["archi_params"]:
        raise ValueError("The \"model_type\" entry in the config.yaml file is invalid; \
            the value of \"model_type\" must match one of the \"archi_params\" keys.")
    
    # Check for tuner-type mismatch
    if model_params["model_type"] not in model_params["archi_params"]:
        raise ValueError("The \"tuner_configs\" entry in the config.yaml file is invalid; \
            the value of \"model_type\" must match one of the \"archi_params\" keys.")
    

def preprocess_and_train(config: dict, path: str):
    """Runs preprocessing and training of the MISST model"""
    # Verify config
    __validate_yaml_values(config)

    # Preprocesses data
    if config["preprocess_data"]:
        preproc = PreProcessor(path, 
            config["annotations"],
            config["dataset_split"],
            config["balance_ratios"],
            config["channels"],
            config["edf_regex"],
            config["hypnogram_regex"]
        )
        preproc.import_and_preprocess()
        preproc.regroup()
        preproc.group_shuffle()
        preproc.save_len()

    # Defines training parameters
    model_params = config["model_params"]
    
    # Adjusts training parameters with tuned architecture
    if config["load_tuned_archi"]:
        # Loads tuned architecture
        os.chdir(f"{path}data/")
        filename = f"hps_{config['tuner_file_to_load']['tuner_type'].name}\
            _{model_params['model_type'].name}\
            _{config['tuner_file_to_load']['tuned_params'].name}.pkl"
        with open(filename, "rb") as f:
            best_hps = pickle.load(f)
        # Adjusts parameters accordingly
        match config["tuner_file_to_load"]["tuned_params"]:
            case "model":
                model_params["archi_params"][model_params["model_type"]].update(best_hps)
            case "lr":
                model_params.update(best_hps)
            case _:
                msg = "The \"tuned_param\" entry in the YAML configuration file is invalid."
                short_err(msg, ValueError(msg))

    # Runs training according to declared training method
    match config["mode"]: 
        case "PLAIN":
            trainer = ModelTrainer(path, config["export_dir"], config["model_params"])
            trainer.basic_train()
        case "DIST":
            trainer = ModelTrainer(path, config["export_dir"], config["model_params"])
            trainer.dist_train()
        case "TUNER":
            trainer = ModelTrainer(path, config["export_dir"], config["model_params"], config["tuner_params"])
            trainer.tuner_train()
        case "GUI" | "DIST_GUI" | "TUNER_GUI":
            DistributedGUI(path, config["export_dir"], config["mode"])
        case _:
            raise ValueError(f"The \"mode\" entry in the YAML configuration file is invalid.") from None