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
from typing import Any, Callable
from abc import ABC, abstractmethod
from tqdm import tqdm

from utils.datasets import GeneratorDataset, ArrayDataset
from utils.enums import TunerType
    

class BaseTrainer(ABC):
    """A general-purpose framework that all Trainer classes must follow"""

    def __init__(self, path: str, export_dir: str, params: dict[str, Any]):
        """Defines class-level variables that all subclasses can access"""
        # Validate and define path
        if not os.path.isdir(f"{path}{export_dir}/"):
            raise ValueError(f"Path '{path}{export_dir}' does not exist.")
        self.PATH = path
        self.EXPORT_DIR = export_dir
        
        # Validate and define model parameters
        missing = self.__missing_params(params)
        if len(missing) > 0:
            raise ValueError(f"The inputted 'params' variable was missing the following keys: {missing}.")
        self.params = params
        
        # Configure callbacks
        self.callbacks = self._preconfigured_callbacks()
        self.metrics = self._preconfigured_metrics()

    def __missing_params(self, params) -> list[str]:
        """Returns a list of missing parameters from the input"""
        missing = []
        if "epochs" not in params:
            missing.append("epochs")
        if "batch_size" not in params:
            missing.append("batch_size")
        if "learning_rate" not in params:
            missing.append("learning_rate")
        if "optimizer" not in params:
            missing.append("optimizer")
        return missing
    
    def basic_train(self) -> None:
        """Trains the model w/o any additional functionality"""
        data = self._import_data()
        model = self._create_model()
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
    def _create_model(self) -> keras.Model:
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
                if valid_port is None:
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

    def tuner_train(self, tuner_type: TunerType) -> None:
        """Trains the model w/ KerasTuner"""
        # Declares consts for tuner
        GOAL = "val_accuracy"
        DIR = f"{self.PATH}{self.EXPORT_DIR}/tuner_results/"
        
        # Match input word w/ tuner
        match tuner_type:
            case TunerType.HYPERBAND:
                tuner = kt.Hyperband(
                    self._tuner_wrapper,
                    objective=GOAL,
                    max_epochs=self.params["epochs"],
                    factor=3,
                    directory=DIR,
                    project_name="Hyperband"
                )
            case TunerType.BAYESIAN:
                tuner = kt.BayesianOptimization(
                    self._tuner_wrapper,
                    objective=GOAL,
                    max_trials=20,
                    directory=DIR,
                    project_name="Bayesian"
                )

        # Define early stopping callback
        stop_early = {"stop_early": tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)}
        self.callbacks.update(stop_early)

        # Search with specified tuner
        data = self._import_data()
        if data is GeneratorDataset:
            tuner.search(
                x=data.train_gen, 
                epochs=self.params["epochs"], 
                validation_data=(data.val_gen), 
                callbacks=list(self.callbacks.values())
            )
        elif data is ArrayDataset:
            tuner.search(
                x=data.x_train, y=data.y_train, 
                epochs=self.params["epochs"], 
                validation_data=(data.x_test, data.y_test), 
                callbacks=list(self.callbacks.values())
            )

        # Save dictionary of optimal hyperparameters
        os.chdir(f"{self.PATH}{self.EXPORT_DIR}/")
        best_hps = tuner.get_best_hyperparameters()[0].values
        BAD_KEYS = ["tuner/epochs", "tuner/initial_epoch", "tuner/bracket", "tuner/round"]
        for key in BAD_KEYS: 
            del best_hps[key]
        with open("best_hyperparams.pkl", "wb") as f: #TODO MAKE THIS BETTER
            pickle.dump(best_hps, f, protocol=pickle.HIGHEST_PROTOCOL)

    @abstractmethod
    def _tuner_wrapper(self, hp: kt.HyperParameters) -> keras.Model:
        """
        Acts as a wrapper for handling the passing of information 
        between the "hp" object and the "_create_model" method. 
        """
        pass
