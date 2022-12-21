#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""main_model.py: Trains the main Multivariate SDCC-BiLSTM Model"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import config
import time
import keras
import keras.backend as K
import numpy as np
import os
from overrides import override
from keras.layers import (
    Conv1D,
    Lambda,
    Reshape,
    LSTM,
    Bidirectional,
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
    

class DataGenerator(keras.utils.Sequence):
    """Sequentially loads saved preprocessed data"""
    
    # Current index in list of all_rec
    cur_rec = 0 
    
    # Create list of all loaded samples
    segments = []
    annots = []

    @override
    def __init__(self, path, batch_size, mode):
        """Initialize global vars"""
        self.PATH = path
        self.BATCH_SIZE = batch_size

        # Change cwd
        os.chdir(f"{path}/08 Other files/{mode}/")
    
    @override
    def __len__(self):
        """Returns the number of batches in one epoch"""
        # Find all files
        all_recs = os.listdir()

        # Iterate through recordings and sums length of each one
        total_len = 0
        for filename in all_recs:
            new_rec = np.load(filename)
            segments = new_rec["annots"].shape[0]
            total_len += segments
        
        # Find floor of total segments over num segments per batch
        return total_len // self.BATCH_SIZE
    
    @override
    def __getitem__(self, idx):
        """Receives batch num and returns batch"""
        # Find all files
        all_recs = os.listdir()

        # Check if new data needs to be loaded
        if self.BATCH_SIZE > len(self.segments):
            filename = all_recs[self.cur_rec]
            new_rec = np.load(filename)
            x = new_rec["x_norm"]
            y = new_rec["annots"]
            self.segments.extend(x)
            self.annots.extend(y)
            self.cur_rec += 1

        # Obtain data slice to return
        slice_seg = np.transpose(np.array(self.segments[:self.BATCH_SIZE])[:, :, :, np.newaxis], axes=(1, 0, 2, 3))
        slice_seg = [i for i in slice_seg] # Convert to list
        slice_annots = np.array(self.annots[:self.BATCH_SIZE])

        # Dispose of old data (start will always be zero)
        self.segments = self.segments[self.BATCH_SIZE:]
        self.annots = self.annots[self.BATCH_SIZE:]

        return slice_seg, slice_annots

    @override
    def on_epoch_end(self):
        """Reset all class variables"""
        self.cur_rec = 0 
        self.segments = []
        self.annots = []
            

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

    def __convert_optimizer(self, optimizer, learning_rate):
        """Creates an optimizer object for an optimizer string"""
        match optimizer:
            case "sgd":
                return keras.optimizers.SGD(learning_rate=learning_rate)
            case "rmsprop":
                return keras.optimizers.RMSprop(learning_rate=learning_rate)
            case "adam":
                return keras.optimizers.Adam(learning_rate=learning_rate)
            case "adadelta":
                return keras.optimizers.Adadelta(learning_rate=learning_rate)
            case "adagrad":
                return keras.optimizers.Adagrad(learning_rate=learning_rate)
            case "adamax":
                return keras.optimizers.Adamax(learning_rate=learning_rate)
            case "nadam":
                return keras.optimizers.Nadam(learning_rate=learning_rate)
            case "ftrl":
                return keras.optimizers.Ftrl(learning_rate=learning_rate)

    @override
    def _import_data(self): 
        """Creates two generators, one for training and one for validation"""
        train_gen = DataGenerator(self.PATH, self.params["batch_size"], "TRAIN")
        val_gen = DataGenerator(self.PATH, self.params["batch_size"], "VAL")
        return [train_gen, val_gen]

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

        # Create list of dilations
        KERNEL = 2 # Non-configurable since this was already tested and optimized by WaveNet
        dilations = [KERNEL ** j for _ in range(self.params["conv_layers"]) for j in range(self.params["sdcc_blocks"])]

        # Defining CNN inputs & layers (Multivariate implementation of WaveNet)
        inputs = []
        convs = []
        for _ in range(NUM_CHANNELS):
            # Create new input and append it to list
            inputs.append(keras.Input(batch_input_shape=(batch_size, int(RECORDING_LEN * SAMPLE_RATE), 1)))
            # Create new conv layers based on dilations
            for layer, rate in enumerate(dilations):
                if layer == 0:
                    conv = Conv1D(filters=self.params["filters"], kernel_size=KERNEL, activation="relu", 
                        padding="causal", dilation_rate = rate)(inputs[-1])
                else:
                    conv = Conv1D(filters=self.params["filters"], kernel_size=KERNEL, activation="relu", 
                        padding="causal", dilation_rate = rate)(conv)
            # Append last convolution
            convs.append(conv)
    
        # Vertically stacks inputs
        vstack = Lambda(lambda a: K.stack(a, axis=2))(convs)
        reshape = Reshape(target_shape=(
            vstack.shape.as_list()[1], 
            vstack.shape.as_list()[2] * vstack.shape.as_list()[3]
        ))(vstack)
        
        # Uses Stacked-LSTM structure
        lstm1 = Bidirectional(LSTM(units=self.params["lstm_nodes"], return_sequences=True))(reshape)
        lstm2 = Bidirectional(LSTM(units=self.params["lstm_nodes"]))(lstm1)

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
        # Create generators
        train_gen = data[0]
        val_gen = data[1]
    
        # Declare vars for parameters of model.fit
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(self.PATH + "08 Other Files/")
        optimizer = self.__convert_optimizer(self.params["optimizer"], 1e-3)

        # Run training w/ or w/o GUI
        if self.gui_objs is not None:
            gui_callback = GUICallback(self.gui_objs, self.params)
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                metrics=["accuracy", self.callbacks.pred_metric])
            model.fit(x=train_gen, validation_data=(val_gen), 
                epochs=self.params["epochs"], callbacks=[gui_callback, model_checkpoint_callback])
        else:
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
                metrics=["accuracy"])
            model.fit(x=train_gen, validation_data=(val_gen), 
                epochs=self.params["epochs"], callbacks=[model_checkpoint_callback])
    

if __name__ == "__main__":
    """Trains the model on the preprocessor.py data"""
    params = {
        "epochs": 100,
        "batch_size": 4,
        "filters": 10,
        "conv_layers": 6,
        "sdcc_blocks": 1,
        "lstm_nodes": 50,
        "dense_nodes": 500,
        "optimizer": "adam"
    }
    
    # Runs training according to declared training method
    match config.MODE:
        case "PLAIN":
            trainer = ModelTrainer(config.PATH, params)
            trainer.driver(False)
        case "DIST":
            trainer = ModelTrainer(config.PATH, params)
            trainer.driver(True)
        case "GUI":
            DistributedGUI(config.PATH, False)
        case "DIST GUI":
            DistributedGUI(config.PATH, True)
        case other:
            raise ValueError(f"Variable \"MODE\" is invalid, got val \"{config.MODE}\"")