#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""main_model.py: Trains the main Multivariate SDCC-BiLSTM Model"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import config
import time
import keras
import random
import os
import datetime
import pickle
import numpy as np
import keras_tuner as kt
from dataclasses import dataclass
from overrides import override
from contextlib import redirect_stdout
from keras.layers import (
    Conv1D,
    Concatenate,
    LSTM,
    Bidirectional,
    Dense,
    Add,
    ReLU,
    BatchNormalization,
    Dropout
)

from gui import GenericGUI
from preprocessor import PreProcessor
from trainers import (
    DistributedTrainer, 
    TunerTrainer,
    GeneratorDataset,
    ArrayDataset
)


class DistributedGUI(GenericGUI):
    """
    Subclasses the GenericGUI to include methods necessary for distributed computing.
    This class specifically references ModelTrainer, which, in the case of a general framework,
    is undesired, hence why the GUI is split into two classes.
    """

    @override
    def __init__(self, path, mode, tuner_type):
        """If dist_comp = True, distributed computing will be used"""
        self.MODE = mode
        self.TUNER_TYPE = tuner_type
        super().__init__(path)

    @override
    def _train_model(self, gui_callback, params):
        """Trains model based on specified mode"""
        trainer = ModelTrainer(self.path, params)
        trainer.set_callbacks(gui_callback={"gui_callback": gui_callback})
        match self.MODE:
            case "GUI":
                trainer.basic_train()
            case "DIST GUI":
                trainer.dist_train()
            case "TUNER GUI":
                trainer.tuner_train(self.tuner_type)
    

class DataGenerator(keras.utils.Sequence):
    """Sequentially loads saved preprocessed data"""

    @override
    def __init__(self, path, batch_size, split):
        """Initialize global vars"""
        self.PATH = path
        self.BATCH_SIZE = batch_size
        self.SPLIT = split

        # Current index in list of all_rec
        self.cur_rec = 0 
        
        # Create list of all loaded samples
        self.x = []
        self.y = []

        # Prevent re-loading of same info
        self.num_batches = None

        # Find all files
        os.chdir(f"{self.PATH}08 Other files/Split/{self.SPLIT}/")
        self.all_recs = os.listdir()
        random.shuffle(self.all_recs)
    
    @override
    def __len__(self):
        """Returns the number of batches in one epoch"""
        # Only runs when var is not already defined
        if self.num_batches == None:
            os.chdir(f"{self.PATH}08 Other files/")
            with open("split_lens.pkl", "rb") as f:
                split_lens = pickle.load(f)
            num_segments = split_lens[self.SPLIT]
            self.num_batches = (num_segments // self.BATCH_SIZE) - 1
        return self.num_batches
    
    @override
    def __getitem__(self, idx):
        """Receives batch num and returns batch"""
        # Change cwd
        os.chdir(f"{self.PATH}08 Other files/Split/{self.SPLIT}/")

        # Check if new data needs to be loaded
        if self.BATCH_SIZE > len(self.x):
            # Updates x and y
            filename = self.all_recs[self.cur_rec]
            new_rec = np.load(filename)
            x_group, y_group = new_rec["x"], new_rec["y"]
            self.x.extend(x_group)
            self.y.extend(y_group)
            self.cur_rec += 1

        # Obtain data slice to return
        slice_x = np.transpose(np.array(self.x[:self.BATCH_SIZE])[:, :, :, np.newaxis], axes=(1, 0, 2, 3))
        slice_x = [i for i in slice_x] # Convert to list
        slice_y = np.array(self.y[:self.BATCH_SIZE])

        # Dispose of old data (start will always be zero)
        self.x = self.x[self.BATCH_SIZE:]
        self.y = self.y[self.BATCH_SIZE:]

        return slice_x, slice_y

    @override
    def on_epoch_end(self):
        """Reset all class variables"""
        self.cur_rec = 0 
        self.x = []
        self.y = []


class ModelTrainer(DistributedTrainer, TunerTrainer):
    """Creates and Trains Model"""

    def __init__(self, path, params):
        """Calls the superclass constructor with 'export_dir'"""
        super().__init__(path, "08 Other files", params)

    @override
    def _preconfigured_callbacks(self) -> dict[str, keras.callbacks.Callback]:
        """Defines preconfigured callbacks"""
        # Define TensorBoard callback
        tensorboard_dir = self.PATH + "08 Other files/TensorBoard/"
        if not os.path.exists(tensorboard_dir):
            os.mkdir(tensorboard_dir)
        log_dir = tensorboard_dir + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        return {"tensorboard_callback": tensorboard_callback}

    def __convert_optimizer(self, optimizer, learning_rate):
        """Returns the corresponding object for a given optimizer name"""
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
    def _import_data(self) -> GeneratorDataset | ArrayDataset: 
        """Creates two generators, one for training and one for validation"""
        train_gen = DataGenerator(self.PATH, self.params["batch_size"], "TRAIN")
        val_gen = DataGenerator(self.PATH, self.params["batch_size"], "VAL")
        return GeneratorDataset(train_gen, val_gen)

    @override
    def _tuner_wrapper(self, hp: kt.HyperParameters) -> keras.Model:
        """Sets the model parameters according to the KerasTuner 'hp' obj"""
        TUNE_MODE = "MODEL" # "LR" (learning rate), "MODEL" (model architecture)
        match TUNE_MODE:
            case "MODEL":
                self.params["filters"]      = hp.Int("filters",      min_value=1,  max_value=10,  step=1  )
                self.params["conv_layers"]  = hp.Int("conv_layers",  min_value=1,  max_value=10,  step=1  )
                self.params["sdcc_blocks"]  = hp.Int("sdcc_blocks",  min_value=1,  max_value=4,   step=1  )
                self.params["lstm_nodes"]   = hp.Int("lstm_nodes",   min_value=10, max_value=200, step=10 )
                self.params["lstm_layers"]  = hp.Int("lstm_layers",  min_value=1,  max_value=5,   step=1  )
                self.params["dense_nodes"]  = hp.Int("dense_nodes",  min_value=20, max_value=500, step=20 )
                self.params["dense_layers"] = hp.Int("dense_layers", min_value=1,  max_value=5,   step=1  )
            case "LR":
                self.params["learning_rate"] = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        
        # Create model
        model = self._create_model()
        optimizer = self.__convert_optimizer(self.params["optimizer"], self.params["learning_rate"])
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=["accuracy"])

        return model

    @override
    def _create_model(self) -> keras.Model:
        """Returns a Keras model, uses strategy to determine batch_size."""
        
        start = time.time()
        print("Started Model Creation")

        # Use preprocessor to get constants
        preproc = PreProcessor(self.PATH)
        RECORDING_LEN = preproc.RECORDING_LEN
        SAMPLE_RATE = preproc.get_edf_info(preproc.import_example_edf())
        NUM_CLASSES = len(preproc.ANNOTATIONS)

        # Determining batch size
        batch_size = int(self.params["batch_size"])

        # Create list of dilations
        KERNEL = 2 # Non-configurable since this was already tested and optimized by WaveNet
        dilations = [KERNEL ** i for i in range(self.params["conv_layers"])]

        # Defining CNN inputs & layers (Multivariate implementation of WaveNet)
        inputs = []
        resnet_outputs = []
        for _ in range(len(preproc.CHANNELS)):
            # Create new input and append it to list
            inputs.append(keras.Input(batch_input_shape=(batch_size, int(RECORDING_LEN * SAMPLE_RATE), 1)))
            
            # Creates multiple SDCC blocks connected w/ residual connections
            for block in range(self.params["sdcc_blocks"]):
                for ind, rate in enumerate(dilations):
                    if block == 0 and ind == 0:
                        normalize = BatchNormalization()(inputs[-1])
                    elif ind == 0:
                        normalize = BatchNormalization()(residual)
                    else:
                        normalize = BatchNormalization()(conv)

                    relu = ReLU()(normalize)
                    dropout = Dropout(0.1)(relu)
                    conv = Conv1D(filters=self.params["filters"], kernel_size=KERNEL, activation="relu", 
                        padding="causal", dilation_rate = rate)(dropout)
                residual = Add()([conv_old, conv]) if block != 0 else Add()([inputs[-1], conv])
                conv_old = conv
            
            # Extra layers (helps w/ performance)
            normalize = BatchNormalization()(residual)
            relu = ReLU()(normalize)

            # Append last layer to list of outputs
            resnet_outputs.append(relu)
    
        # Concatenate inputs
        merged = Concatenate(axis=2)(resnet_outputs)
        
        # Uses Stacked-LSTM structure
        for layer in range(self.params["lstm_layers"]):
            if layer == 0 and self.params["lstm_layers"] > 1:
                lstm = Bidirectional(LSTM(units=self.params["lstm_nodes"], return_sequences=True))(merged)
            elif layer == 0:
                lstm = Bidirectional(LSTM(units=self.params["lstm_nodes"]))(merged)
            elif layer == self.params["lstm_layers"] - 1:
                lstm = Bidirectional(LSTM(units=self.params["lstm_nodes"]))(lstm)
            else:
                lstm = Bidirectional(LSTM(units=self.params["lstm_nodes"], return_sequences=True))(lstm)

        # Dense layers
        for layer in range(self.params["dense_layers"]):
            dropout = Dropout(0.5)(lstm if layer == 0 else dense) 
            dense = Dense(units=self.params["dense_nodes"], activation="relu")(dropout)
        
        # Output
        output = Dense(units=NUM_CLASSES, activation="softmax")(dense)

        # Create Model
        model = keras.Model(inputs=inputs, outputs=output)

        # Save plot of model
        os.chdir(f"{self.PATH}mouse_psg/")
        keras.utils.plot_model(model, to_file="Keras_Model_Plot.png", show_shapes=True)

        # Print out summary of model to txt
        with open("model_summary.txt", "w") as f:
            with redirect_stdout(f):
                model.summary()

        # Print elapsed time
        elapsed = time.time() - start
        print(f"Finished creating model || Elapsed time: {elapsed}s")

        return model

    @override
    def _train_model(self, model, data) -> None:
        """
        Trains the model by running model.fit(). A vanilla custom training loop will
        not work with distributed training. Any variables used by model.fit must be
        declared and initialized within this scope
        """
    
        # Set optimizer
        optimizer = self.__convert_optimizer(self.params["optimizer"], self.params["learning_rate"])

        # Set metrics depending on whether GUI is enabled or not
        if "gui_callback" in self.callbacks:
            metrics = ["accuracy", self.callbacks["gui_callback"].pred_metric]
        else:
            metrics = ["accuracy"]

        # Run training w/ or w/o GUI
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy',
            metrics=metrics)
        model.fit(x=data.train_gen, validation_data=(data.val_gen), epochs=self.params["epochs"], 
            callbacks=list(self.callbacks.values()))

        model.save(f"{self.PATH}08 Other files/")
    

if __name__ == "__main__":
    """Trains the model on the preprocessor.py data"""
    # Define training parameters
    TUNER_TYPE = "Hyperband" # "Hyperband", "Bayesian" 
    LOAD_FROM_TUNER = True # Default is True, set as False to manually configure "params" var
    params = {
        # General Hyperparameters
        "epochs":            50,
        "batch_size":        16,
        "learning_rate": 3.2e-4,
        "optimizer":     "adam",
        # SDCC
        "filters":            6,
        "conv_layers":        5,
        "sdcc_blocks":        2,
        # BiLSTM
        "lstm_nodes":       200,
        "lstm_layers":        2,
        # Dense
        "dense_nodes":      320,
        "dense_layers":       1,
    }
    if LOAD_FROM_TUNER:
        os.chdir(f"{config.PATH}08 Other files/")
        with open("best_hyperparams.pkl", "rb") as f:
            best_hps = pickle.load(f)
        params.update(best_hps)
    
    # Runs training according to declared training method
    match config.MODE:
        case "PLAIN":
            trainer = ModelTrainer(config.PATH, params)
            trainer.basic_train()
        case "DIST":
            trainer = ModelTrainer(config.PATH, params)
            trainer.dist_train()
        case "TUNER":
            trainer = ModelTrainer(config.PATH, params)
            trainer.tuner_train(TUNER_TYPE)
        case "GUI" | "DIST GUI" | "TUNER GUI":
            DistributedGUI(config.PATH, config.MODE, TUNER_TYPE)
        case other:
            raise ValueError(f"Variable \"MODE\" is invalid, got val \"{config.MODE}\"")
