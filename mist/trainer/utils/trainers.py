#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
trainers.py: Defines the abstract classes inherited by the ModelTrainer class. 
Also defines classes used for interfacing with Trainer classes. This module is 
separate from other files, and can be used as an API/library.
"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import tensorflow as tf
import keras
import contextlib
import socket
import pickle
import keras_tuner as kt
import keras
import os

from enum import Enum
from typing import Callable
from abc import ABC, abstractmethod
from tqdm import tqdm
from overrides import override

from .req_params import ModelParams, TunerParams
from .enum_vals import TunerType
from .datasets import GeneratorDataset, ArrayDataset


class BaseTrainer(ABC):
    """A general-purpose framework that all Trainer classes must follow"""

    def __init__(self, path: str, export_dir: str, params: ModelParams):
        """Defines class-level variables that all subclasses can access"""
        # Validate and define path
        if not os.path.isdir(f"{path}{export_dir}/"):
            raise ValueError(f"Path '{path}{export_dir}' does not exist.")
        self.PATH = path
        self.EXPORT_DIR = export_dir
        
        # Set params var
        self.params = params

        # Configure callbacks
        self.callbacks = self._preconfigured_callbacks()
        self.metrics = self._preconfigured_metrics()

    def basic_train(self) -> None:
        """Trains the model w/o any additional functionality"""
        data = self._import_data()
        model = self._create_model(self.params["model_type"])
        self._train_model(model, data)

    def set_callbacks(self, callbacks: dict[str, keras.callbacks.Callback]) -> None:
        """
        Appends inputted callbacks to pre-existing list of callbacks. This allows for
        scope-dependent callbacks to be created externally.
        """
        self.callbacks.update(callbacks)

    @abstractmethod
    def _preconfigured_callbacks(self) -> dict[str, keras.callbacks.Callback]:
        """
        Returns a list of predefined callbacks. Scope-independent callbacks 
        should be initialized here
        """
        pass

    def set_metrics(self, metrics: dict[str, str | keras.metrics.Metric | Callable]):
        """
        Appends inputted metric to pre-existing list of metrics. This allows for
        scope-dependent metrics to be created externally
        """
        self.metrics.update(metrics)

    @abstractmethod
    def _preconfigured_metrics(self) -> dict[str, str | keras.metrics.Metric | Callable]:
        """
        Returns a list of predefined metrics. Scope-independent metrics
        should be initialized here.
        """
        pass

    @abstractmethod
    def _import_data(self) -> GeneratorDataset | ArrayDataset:
        """Returns a dataset obj that can be fed directly into '_train_model'"""
        pass
    
    @abstractmethod
    def _create_model(self, model_type: Enum) -> keras.Model:
        """Returns a Keras.Model instance that can be fed directly into '_train_model'"""
        pass

    @abstractmethod
    def _train_model(self, model: keras.Model, data: GeneratorDataset | ArrayDataset) -> None:
        """Trains the model by running model.fit()"""
        pass


class DistributedTrainer(BaseTrainer, ABC):
    """A general-purpose framework for distributed training"""

    # Config vars (MUST BE CHANGED PER USER)
    DEVICE_IPS = {
        "chief": ["192.168.1.175"],
        "worker": ["192.168.1.175", "192.168.1.175"] #TODO replace w/ actual device ips
    }
    CURRENT_NODE = {"type": "chief", "index": 0}

    def _is_port_open(self, ip: str, port: str) -> bool:
        """Tests if 'port' at 'ip' is open by attempting to bind; uses contextlib for Automatic Resouce Management"""
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind((ip, port))
            except socket.error:
                return False
            return True
 
    def _generate_config(self) -> dict:
        """Generates a tf_config variable for MultiWorkerMirroredStrategy"""
        # Config vars
        PORT_START = 10000 # Avoids "Well-Known Ports"
        PORT_LIM = 10050 # Highest number port to search to

        # Check every IP and find which ports work
        valid_ports = {node_type: [] for node_type in self.DEVICE_IPS}
        for node_type in self.DEVICE_IPS:
            for ip in tqdm(self.DEVICE_IPS[node_type], desc=f"Finding ports for {node_type}"):
                valid_port = None
                for port in range(PORT_START, PORT_LIM):
                    if self._is_port_open(ip, port):
                        valid_port = port
                        break
                if valid_port == None:
                    raise RuntimeError(f"Could not find a valid port for ip: \"{ip}\".")
                valid_ports[node_type].append(valid_port)

        # Merge ips and valid ports
        cluster = {node_type: [] for node_type in self.DEVICE_IPS}
        for node_type in self.DEVICE_IPS:
            for index in range(len(self.DEVICE_IPS[node_type])):
                cluster[node_type].append(f"{self.DEVICE_IPS[node_type][index]}:{valid_ports[node_type][index]}")

        # Initializes TF_CONFIG
        tf_config = {
            "cluster": cluster,
            "task": self.CURRENT_NODE
        }
        return tf_config

    def dist_train(self) -> None:
        """Trains the model w/ Distributed Computing"""
        # Enable distributed computing and prepare necessary vars
        self.dist_comp = True
        tf_config = self._generate_config()
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        self.strategy = strategy

        # Run distributed instances
        with strategy.scope():
            self.params["batch_size"] *= int(self.strategy.num_replicas_in_sync)
            self._basic_train()


class TunerTrainer(BaseTrainer, ABC):
    """A general-purpose framework for integrating KerasTuner into models"""

    @override
    def __init__(self, path: str, export_dir: str, params: ModelParams, tuner_params: TunerParams = None):
        """Includes the tuner_params variable as part of the constructor's arguments"""
        # Runs the normal BaseTrainer class constructor
        super().__init__(path, export_dir, params)
        
        # Defines tuner_param var
        self.tuner_params = tuner_params

    def tuner_train(self) -> None:
        """Trains the model w/ KerasTuner"""
        # Verify that tuner_params has been entered
        if self.tuner_params == None:
            raise ValueError(
                """
                TunerTrainer's 'tuner_train()' method cannot be executed without the 'tuner_params' variable. 
                The 'tuner_params' variable should be entered in the constructor of TunerTrainer
                """
            )

        # Ensure that a valid goal is picked
        PREFIXES = ["val_"]
        stripped = self._remove_prefixes(self.tuner_params["goal"], PREFIXES)
        valid_goal = any(stripped == metric.name for metric in self.metrics.values())
        if not valid_goal:
            raise ValueError(
                f"""
                'goal' in 'tuner_params' must match with a metric; got goal: '{self.tuner_params['goal']}',
                but the only metrics were: {[metric.name for metric in self.metrics.values()]}.
                """
            )

        # Declares consts for tuner
        dir_path = f"{self.PATH}{self.EXPORT_DIR}/{self.tuner_params['dir_name']}/"
        
        # Match input word w/ tuner
        match self.tuner_params["tuner_type"]:
            case TunerType.HYPERBAND:
                tuner_config = self.tuner_params["tuner_configs"][TunerType.HYPERBAND]
                tuner = kt.Hyperband(
                    self.__tuner_wrapper,
                    objective=self.tuner_params["goal"],
                    max_epochs=tuner_config["max_epochs"],
                    factor=tuner_config["factor"],
                    directory=dir_path,
                    project_name="Hyperband"
                )
            case TunerType.BAYESIAN:
                tuner_config = self.tuner_params["tuner_configs"][TunerType.BAYESIAN]
                tuner = kt.BayesianOptimization(
                    self.__tuner_wrapper,
                    objective=self.tuner_params["goal"],
                    max_trials=self.tuner_config["max_trials"],
                    directory=dir_path,
                    project_name="Bayesian"
                )

        # Define early stopping callback
        stop_early = {"stop_early": tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)}
        self.callbacks.update(stop_early)

        # Search with specified tuner
        data = self._import_data()
        if type(data) == GeneratorDataset:
            tuner.search(
                x=data.train_gen,
                epochs=self.params["epochs"], 
                validation_data=(data.val_gen), 
                callbacks=list(self.callbacks.values())
            )
        elif type(data) == ArrayDataset:
            tuner.search(
                x=data.x_train, y=data.y_train, 
                epochs=self.params["epochs"], 
                validation_data=(data.x_test, data.y_test), 
                callbacks=list(self.callbacks.values())
            )

        # Process optimized hyperparameters
        BAD_KEYS = ["tuner/epochs", "tuner/initial_epoch", "tuner/bracket", "tuner/round"]
        best_hps = tuner.get_best_hyperparameters()[0].values
        for key in BAD_KEYS: 
            del best_hps[key]

        # Save file of best params
        os.chdir(f"{self.PATH}{self.EXPORT_DIR}/")
        filename = f"hps_{self.tuner_params['tuner_type'].name}_{self.params['model_type'].name}_{self.tuner_params['params_to_tune'].name}.pkl"
        with open(filename, "wb") as f:
            pickle.dump(best_hps, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _remove_prefixes(self, main_str: str, prefixes: list[str]) -> str:
        """
        Checks a list of prefixes and, upon finding a single match, 
        deletes the prefix from a string. A prefix must be at the very 
        start of the string (index zero)
        """
        for prefix in prefixes:
            if main_str[:len(prefix)] == prefix:
                main_str = main_str[len(prefix):]
                break
        return main_str

    def __tuner_wrapper(self, hp: kt.HyperParameters) -> keras.Model:
        """Passes the tuner_params["param_to_tune"] variable to the _create_model_wrapper() method, thus preserving encapsulation"""
        model = self._model_creator_wrapper(self.params["model_type"], hp, self.tuner_params["params_to_tune"])
        return model

    @abstractmethod
    def _model_creator_wrapper(self, model_type: Enum, hp: kt.HyperParameters, param_to_tune: Enum) -> keras.Model:
        """
        Acts as a wrapper for handling the passing of information 
        between the "hp" object and the "_create_model" method. 
        The datatype of "param_to_tune" is entirely dependent on the
        datatype of the value entered into the "tuner_params" variable.
        """
        pass
