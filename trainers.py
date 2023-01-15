#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""trainers.py: Defines the abstract classes inherited by the ModelTrainer class"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import tensorflow as tf
import keras
import contextlib
import socket
import keras_tuner as kt
import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from typing import Callable


class BaseTrainer(ABC):
    """A general-purpose framework that all Trainer classes must follow"""

    def __init__(self, path: str):
        """Ensures that all subclasses can access the path"""
        self.PATH = path
        self.callbacks = self._preconfigured_callbacks()

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

    @abstractmethod
    def _import_data(self) -> list[Callable | np.ndarray | list]:
        """Returns a data obj that can be fed directly into '_train_model'"""
        pass
    
    @abstractmethod
    def _create_model(self) -> keras.Model:
        """Returns a Keras.Model instance that can be fed directly into '_train_model'"""
        pass

    @abstractmethod
    def _train_model(self, model: keras.Model, data: list[Callable | np.ndarray | list]) -> None:
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
    dist_comp = False

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
            self.basic_train()


class TunerTrainer(BaseTrainer, ABC):
    """A general-purpose framework for integrating KerasTuner into models"""

    def tuner_train(self, tuner_type: str) -> None:
        """Trains the model w/ KerasTuner"""
        # Declares consts for tuner
        GOAL = "val_accuracy"
        DIR = self.PATH + "08 Other files/Tuner Results/"
        
        # Match input word w/ tuner
        match tuner_type:
            case "Hyperband":
                tuner = kt.Hyperband(
                    self._wrapper,
                    objective=GOAL,
                    max_epochs=10,
                    factor=3,
                    directory=DIR,
                    project_name="Hyperband"
                )
            case "Bayesian":
                tuner = kt.BayesianOptimization(
                    self._wrapper,
                    objective=GOAL,
                    max_trials=10,
                    directory=DIR,
                    project_name="Bayesian"
                )

        # Define early stopping callback
        stop_early = {"stop_early": tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)}
        self.callbacks.update(stop_early)

        # Search with specified tuner
        train_gen, val_gen = self._import_data()
        tuner.search(x=train_gen, epochs=50, validation_data=(val_gen), callbacks=list(self.callbacks.values()))

        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Print summary
        tuner.results_summary(num_trials=10)

    @abstractmethod
    def _wrapper(self, hp: kt.HyperParameters) -> keras.Model:
        """
        Acts as a wrapper for handling the passing of information 
        between the "hp" object and the "_create_model" method. 
        """
        pass
