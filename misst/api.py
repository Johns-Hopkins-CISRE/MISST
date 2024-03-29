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
        msg = "The \"model_type\" entry in the config.yaml file is invalid; \
            the value of \"model_type\" must match one of the \"archi_params\" keys."
        short_err(msg, ValueError(msg))

    # Check for tuner-type mismatch
    if model_params["model_type"] not in model_params["archi_params"]:
        msg = "The \"tuner_configs\" entry in the config.yaml file is invalid; \
            the value of \"model_type\" must match one of the \"archi_params\" keys."
        short_err(msg, ValueError(msg))

def preprocess_and_train(config: dict, path: str):
    """Runs preprocessing and training of the MISST model"""
    # Verify config
    __validate_yaml_values(config)

    # Checks if pre-processed data already exists
    if config["override_existing_preprocessed_data"] or not os.path.exists(f"{path}data/preprocessed/"):
        preproc = PreProcessor(path,
            config["epoch_len"],
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
        preproc.clear_dirs()

    # Defines training parameters
    model_params = config["model_params"]
    
    # Adjusts training parameters with tuned architecture
    if config["load_tuned_archi"]:
        # Loads tuned architecture
        os.chdir(f"{path}data/")
        filename = f"hps_{config['tuner_file_to_load']['tuner_type'].name}\
            _{model_params['model_type'].name}\
            _{config['tuner_file_to_load']['tuned_params'].name}.pkl"
        try:
            with open(filename, "rb") as f:
                best_hps = pickle.load(f)
        except FileNotFoundError as err:
            msg = ("The file containing tuned model parameters could not be found. This file " +
                "is created after the MISST tuner is executed, and can be found in the \"data\" " +
                "directory.")
            short_err(msg, err)
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
            msg = f"The \"mode\" entry in the YAML configuration file is invalid."
            short_err(msg, ValueError(msg))
