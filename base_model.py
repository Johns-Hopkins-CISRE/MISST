#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""base_model.py: Trains the main Multivariate SDCC-BiLSTM Model"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import config
import time
import keras
import keras.backend as K
import joblib
from overrides import override
from keras.layers import (
    Conv1D,
    MaxPooling1D,
    Lambda,
    Reshape,
    LSTM,
    Dense
)

from gui import GenericGUI, GUICallback
from preprocessor import PreProcessor
from distributed_trainer import DistributedTrainer


class DistributedGUI(GenericGUI):
    """
    Subclasses the GenericGUI to include methods necessary for distributed computing.
    This class specifically references ModelTrainer, which, in the case of a general framework,
    is undesired, hence why the GUI is split into two classes.
    """

    @override
    def __init__(self, path, dist_comp):
        """If dist_comp = True, distributed computing will be used"""
        super().__init__(path)
        self.dist_comp = dist_comp

    @override
    def _train_model(self, gui_objs, params):
        trainer = ModelTrainer(self.path, params)
        trainer.enable_gui(gui_objs)
        trainer.driver(self.dist_comp)
    

class ModelTrainer(DistributedTrainer):
    """Creates and Trains Model"""
    
    gui_objs = None

    def __init__(self, path, params):
        """Initializes class level variables, params is just user inputted values"""
        self.PATH = path
        self.params = params

    def enable_gui(self, gui_objs):
        """
        Enables callbacks while still giving the option to train separate of GUI.
        Only takes in gui_objs instead of GUICallback since all callbacks must be initialized
        inside 'train_model'
        """
        self.gui_objs = gui_objs

    @override
    def _import_data(self):
        joblib.load(self.PATH + "01 Raw Data/processed_data.joblib")

    @override
    def _create_model(self):
        """Returns a Keras model, uses strategy to determine batch_size"""
        start = time.time()
        print("Started Model Creation")

        # Use preprocessor to get constants
        preproc = PreProcessor(self.PATH)
        RECORDING_LEN = preproc.RECORDING_LEN
        SAMPLE_RATE, NUM_CHANNELS = preproc.get_edf_info(preproc.import_example_edf())
        NUM_CLASSES = len(preproc.ANNOTATIONS)

        # Determining batch size
        if self.dist_comp:
            batch_size = int(self.params["batch_size"] * self.strategy.num_replicas_in_sync)
        else:
            batch_size = int(self.params["batch_size"])

        # Defining CNN inputs & layers (Multivariate implementation of WaveNet)
        inputs = []
        pool = []
        for _ in range(0, NUM_CHANNELS):
            inputs.append(keras.Input(batch_input_shape=(batch_size, int(RECORDING_LEN * SAMPLE_RATE), 1)))
            conv = Conv1D(filters=self.params["filters"], kernel_size=10, strides=2, activation="relu", padding="same")(inputs[-1])
            pool.append(MaxPooling1D(pool_size=100, padding="same")(conv))
        
        # Vertically stacks inputs
        vstack = Lambda(lambda a: K.stack(a, axis=2))(pool)
        reshape = Reshape(target_shape=(
            vstack.shape.as_list()[1], 
            vstack.shape.as_list()[2] * vstack.shape.as_list()[3]
        ))(vstack)
        
        # Uses Stacked-LSTM structure
        lstm1 = LSTM(units=self.params["lstm_nodes"], return_sequences=True)(reshape)
        lstm2 = LSTM(units=self.params["lstm_nodes"])(lstm1)

        # Dense layers & Output
        dense1 = Dense(units=self.params["dense_nodes"], activation="relu")(lstm2)
        dense2 = Dense(units=self.params["dense_nodes"], activation="relu")(dense1)
        output = Dense(units=NUM_CLASSES, activation="softmax")(dense2)

        # Create Model
        model = keras.Model(inputs=inputs, outputs=output)

        # Save plot of model
        keras.utils.plot_model(model, to_file="Keras_Model_Plot.png", show_shapes=True)

        elapsed = time.time() - start
        print(f"Finished creating model || Elapsed time: {elapsed}s")

        return model

    @override
    def _train_model(self, model, data):
        """
        Trains the model by running model.fit(). A vanilla custom training loop will
        not work with distributed training. Any variables used by model.fit must be
        declared and initialized within this scope
        """
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(self.PATH + "08 Other Files/")
        preprocessor = PreProcessor(self.PATH)
        dataset = preprocessor.create_dataset()
        if self.gui_objs is not None:
            callback = GUICallback(self.gui_objs, self.params)
            model.compile(optimizer=self.params["optimizer"], metrics=["accuracy", self.callbacks.pred_metric])
            model.fit(epochs=self.params["epochs"], callbacks=[callback, model_checkpoint_callback])
        else:
            model.compile(optimizer=self.params["optimizer"], metrics=["accuracy"])
            model.fit(epochs=self.params["epochs"], callbacks=[model_checkpoint_callback])
    

if __name__ == "__main__":
    """Trains the model on the preprocessor.py data"""
    params = {
        "epochs": 100,
        "batch_size": 4,
        "filters": 4,
        "lstm_nodes": 50,
        "dense_nodes": 500,
        "optimizers": "adam"
    }
    
    # Runs training according to declared training method
    match config.MODE:
        case "PLAIN":
            trainer = ModelTrainer(config.PATH)
            trainer.driver(False)
        case "DIST":
            trainer = ModelTrainer(config.PATH)
            trainer.driver(True)
        case "GUI":
            DistributedGUI(config.PATH, False)
        case "DIST GUI":
            DistributedGUI(config.PATH, True)
        case other:
            raise ValueError(f"Variable \"MODE\" is invalid, got val \"{MODE}\"")