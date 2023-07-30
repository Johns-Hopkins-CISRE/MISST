#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""train.py: A training program that trains a model based on the configuration file"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import yaml
import os
import misst

# Sets global path as filepath of this example
path = os.path.dirname(os.path.abspath(__file__))

# Changes the path to this example
os.chdir(path)

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Run MISST's trainer
misst.preprocess_and_train(config, path)