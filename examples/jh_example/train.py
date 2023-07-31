#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""train.py: A training program that trains a model based on the configuration file"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import yaml
import os
import misst

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get current path
path = os.getcwd() + "/"

# Run MISST's trainer
misst.preprocess_and_train(config, path)