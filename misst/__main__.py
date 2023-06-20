#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""__main__.py: Selects whether to train or generate predictions based on user input"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import argparse

from misst.global_config import PATH
from misst.trainer.main import preprocess_and_train
from misst.predictor import predictor


# Receives arguments from CLI
parser = argparse.ArgumentParser(description="A Murine Intelligent Sleep Staging Tool")
parser.add_argument("--mode", type=str, required=True, 
    help="Decides whether you'd like to train a new MISST model or use a pretrained model.",
    choices=["train", "generate"]
)
args = parser.parse_args()

# Runs the corresponding module
match args.mode:
    case "train":
        preprocess_and_train(PATH)
    case "generate":
        print("stub")
