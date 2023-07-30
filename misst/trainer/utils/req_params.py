#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""req_params.py: Contains classes representing required params for their associated trainer classes"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

from typing import Any, TypedDict
from enum import Enum


class ModelParams(TypedDict):
    """General Model/Training Hyperparameters; required by all subclasses of BaseTrainer"""
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: str
    model_type: Enum
    archi_params: dict[Enum, dict[str, Any]]


class _HyperbandParams(TypedDict):
    max_epochs: int
    factor: int


class _BayesianParams(TypedDict):
    max_trials: int


class _TunerConfigs(TypedDict):
    hyperband: _HyperbandParams
    bayesian: _BayesianParams


class TunerParams(TypedDict):
    """Parameters for model tuning; required by TunerTrainer"""
    tuner_type: str
    params_to_tune: Enum
    goal: str
    dir_name: str
    tuner_configs: _TunerConfigs
