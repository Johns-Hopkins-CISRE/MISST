#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
distributed_trainer.py: Defines the abstract class "DistributedTrainer," allowing for easily customizable 
distributed computing
"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import tensorflow as tf
import contextlib
import socket
from abc import ABC, abstractmethod
from tqdm import tqdm

class DistributedTrainer(ABC):
    """A general-purpose framework for distributed training"""

    # Config vars (MUST BE CHANGED PER USER)
    DEVICE_IPS = {
        "chief": ["192.168.1.175"],
        "worker": ["192.168.1.175", "192.168.1.175"] #TODO replace w/ actual device ips
    }
    CURRENT_NODE = {"type": "chief", "index": 0}
    dist_comp = False

    def _is_port_open(self, ip, port):
        """Tests if 'port' at 'ip' is open by attempting to bind; uses contextlib for Automatic Resouce Management"""
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            try:
                s.bind((ip, port))
            except socket.error:
                return False
            return True

    def _generate_config(self):
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

    def driver(self, dist_comp):
        """
        Chooses either distributed training or normal training depending on the input. Interally handling 
        all the training steps allows for the privatization of all training methods, and keeps the difference 
        between distributed and normal execution out of the eyes of the user.
        """
        if dist_comp:
            # Enable distributed computing and prepare necessary vars
            self.dist_comp = True
            tf_config = self._generate_config()
            strategy = tf.distribute.MultiWorkerMirroredStrategy()
            self.strategy = strategy
            # Run distributed instances
            with strategy.scope():
                model = self._create_model()
                data = self._import_data()
                self._train_model(model, data)
        else:
            self.dist_comp = False
            model = self._create_model()
            data = self._import_data()
            self._train_model(model, data)
    
    @abstractmethod
    def _import_data(self):
        """Returns a data obj that can be fed directly into '_train_model'"""
        pass
    
    @abstractmethod
    def _create_model(self):
        """Returns a Keras.Model instance that can be fed directly into '_train_model'"""
        pass

    @abstractmethod
    def _train_model(self, model, data):
        """
        Trains the model by running model.fit(). A vanilla custom training loop will
        not work with distributed training. Any variables used by model.fit must be
        declared and initialized within this scope.
        """
        pass