#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""req_params.py: Contains classes representing required params for their associated trainer classes"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

from typing import Any, TypedDict
from enum import Enum

from utils.enum_vals import Optimizers, TunerType


class ModelParams(TypedDict):
    """General Model/Training Hyperparameters; required by all subclasses of BaseTrainer"""
    epochs: int
    batch_size: int
    learning_rate: float
    optimizer: Optimizers
    model_type: Enum
    archi_params: dict[Enum, dict[str, Any]]


class _HyperbandParams(TypedDict):
    factor: int


class _BayesianParams(TypedDict):
    max_trials: int


class _TunerConfigs(TypedDict):
    TunerType.HYPERBAND: _HyperbandParams
    TunerType.BAYESIAN: _BayesianParams


class TunerParams(TypedDict):
    """Parameters for model tuning; required by TunerTrainer"""
    tuner_type: TunerType
    param_to_tune: Enum
    goal: str
    dir_name: str
    tuner_configs: _TunerConfigs
