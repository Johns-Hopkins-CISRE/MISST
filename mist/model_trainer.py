#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""main.py: Trains the main Multivariate SDCC-BiLSTM Model"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import time
import keras
import random
import os
import datetime
import pickle
import numpy as np
import keras_tuner as kt
import tensorflow as tf
from enum import Enum
from overrides import override
from typing import Any, Callable
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

from utils.datasets import GeneratorDataset, ArrayDataset
from utils.enum_vals import Splits
from utils.trainers import DistributedTrainer, TunerTrainer

from project_enums import TrainingModes, ModelType, TuneableParams
from preprocessor import PreProcessor
from gui import GenericGUI, GUICallback


class DistributedGUI(GenericGUI):
    """
    Subclasses the GenericGUI to include methods necessary for distributed computing.
    This class specifically references ModelTrainer, which, in the case of a general framework,
    is undesired, hence why the GUI is split into two classes.
    """

    @override
    def __init__(self, path: str, export_dir: str, mode: TrainingModes):
        """Overrides GenericGUI's init to allow for parameters used by _train_model()"""
        self.MODE = mode
        self.EXPORT_DIR = export_dir
        super().__init__(path)

    @override
    def _train_model(self, gui_objs: dict[str, Any], params: dict[str, Any]):
        """Trains model based on specified mode"""
        # Create instance of ModelTrainer
        trainer = ModelTrainer(self.PATH, self.EXPORT_DIR, params)

        # Create callback and intialize trainer w/ callback & metric
        gui_callback = GUICallback(self.PATH, gui_objs, params)
        trainer.set_callbacks({"gui_callback": gui_callback})
        trainer.set_metrics({"pred_metric": gui_callback.pred_metric})
        
        # Trains based on mode
        match self.MODE:
            case TrainingModes.GUI:
                trainer.basic_train()
            case TrainingModes.DIST_GUI:
                trainer.dist_train()
            case TrainingModes.TUNER_GUI:
                trainer.tuner_train()
    

class DataGenerator(keras.utils.Sequence):
    """Sequentially loads saved preprocessed data"""

    @override
    def __init__(self, path: str, batch_size: int, split: Splits):
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
        os.chdir(f"{self.PATH}data/split/{self.SPLIT}/")
        self.all_recs = os.listdir()
        random.shuffle(self.all_recs)
    
    @override
    def __len__(self):
        """Returns the number of batches in one epoch"""
        # Only runs when var is not already defined
        if self.num_batches == None:
            os.chdir(f"{self.PATH}data/")
            with open("split_lens.pkl", "rb") as f:
                split_lens = pickle.load(f)
            num_segments = split_lens[str(self.SPLIT)]
            self.num_batches = (num_segments // self.BATCH_SIZE) - 1
        return self.num_batches
    
    @override
    def __getitem__(self, idx):
        """Receives batch num and returns batch"""
        # Change cwd
        os.chdir(f"{self.PATH}data/split/{self.SPLIT}/")

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


class LRScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Acts as a learning rate scheduler; callback doesn't work w/ KerasTuner"""
    
    def __init__(self, initial_learning_rate):
        """Saves initial lr"""
        self.initial_learning_rate = initial_learning_rate

    def __call__(self, step):
        """Called at start of each epoch, returns a modified lr"""
        # TODO replace this with warmup and anneal cosine lr
        return self.initial_learning_rate


class ModelTrainer(DistributedTrainer, TunerTrainer):
    """Creates and Trains Model"""

    @override
    def _preconfigured_callbacks(self) -> dict[str, keras.callbacks.Callback]:
        """Defines preconfigured callbacks"""
        # Define TensorBoard callback
        tensorboard_dir = f"{self.PATH}data/TensorBoard/"
        if not os.path.exists(tensorboard_dir):
            os.mkdir(tensorboard_dir)
        log_dir = tensorboard_dir + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        return {"tensorboard_callback": tensorboard_callback}
    
    @override
    def _preconfigured_metrics(self) -> dict[str, str | keras.metrics.Metric | Callable]:
        """Creates default accuracy metric"""
        accuracy = {"accuracy": keras.metrics.SparseCategoricalAccuracy()}
        return accuracy

    def _create_sdcc(self, preproc: PreProcessor, batch_size: int, archi_params: dict[str, Any]) -> keras.Model:
        """Creates an SDCC-BiLSTM"""
        # Use preprocessor for getting constants
        RECORDING_LEN = preproc.RECORDING_LEN
        SAMPLE_RATE = preproc.get_edf_info(preproc.import_example_edf())
        NUM_CLASSES = len(preproc.ANNOTATIONS)

        # Create list of dilations
        KERNEL = 2 # Non-configurable since this was already tested and optimized by WaveNet
        dilations = [KERNEL ** i for i in range(archi_params["conv_layers"])]

        # Defining CNN inputs & layers (Multivariate implementation of WaveNet)
        inputs = []
        resnet_outputs = []
        for _ in range(len(preproc.CHANNELS)):
            # Create new input and append it to list
            inputs.append(keras.Input(batch_input_shape=(batch_size, int(RECORDING_LEN * SAMPLE_RATE), 1)))
            
            # Creates multiple SDCC blocks connected w/ residual connections
            for block in range(archi_params["sdcc_blocks"]):
                for ind, rate in enumerate(dilations):
                    if block == 0 and ind == 0:
                        normalize = BatchNormalization()(inputs[-1])
                    elif ind == 0:
                        normalize = BatchNormalization()(residual)
                    else:
                        normalize = BatchNormalization()(conv)

                    relu = ReLU()(normalize)
                    dropout = Dropout(0.1)(relu)
                    conv = Conv1D(filters=archi_params["filters"], kernel_size=KERNEL, activation="relu", 
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
        for layer in range(archi_params["lstm_layers"]):
            if layer == 0 and archi_params["lstm_layers"] > 1:
                lstm = Bidirectional(LSTM(units=archi_params["lstm_nodes"], return_sequences=True))(merged)
            elif layer == 0:
                lstm = Bidirectional(LSTM(units=archi_params["lstm_nodes"]))(merged)
            elif layer == archi_params["lstm_layers"] - 1:
                lstm = Bidirectional(LSTM(units=archi_params["lstm_nodes"]))(lstm)
            else:
                lstm = Bidirectional(LSTM(units=archi_params["lstm_nodes"], return_sequences=True))(lstm)

        # Dense layers
        for layer in range(archi_params["dense_layers"]):
            dropout = Dropout(0.5)(lstm if layer == 0 else dense) 
            dense = Dense(units=archi_params["dense_nodes"], activation="relu")(dropout)
        
        # Output
        output = Dense(units=NUM_CLASSES, activation="softmax")(dense)

        # Create Model
        model = keras.Model(inputs=inputs, outputs=output)
        
        return model

    def _create_bottleneck(self, preproc: PreProcessor, batch_size: int, archi_params: dict[str, Any]) -> keras.Model:
        """Creates a Bottleneck CNN, architecture is based on Mignot's paper"""
        #TODO add bottleneck cnn
        ...
        
    @override
    def _import_data(self) -> GeneratorDataset | ArrayDataset: 
        """Creates two generators, one for training and one for validation"""
        train_gen = DataGenerator(self.PATH, self.params["batch_size"], Splits.TRAIN)
        val_gen = DataGenerator(self.PATH, self.params["batch_size"], Splits.VAL)
        return GeneratorDataset(train_gen, val_gen)

    @override
    def _model_creator_wrapper(self, model_type: Enum, hp: kt.HyperParameters, param_to_tune: Any) -> keras.Model:
        """Sets the model parameters according to the KerasTuner 'hp' obj"""
        match param_to_tune:
            case TuneableParams.MODEL:
                archi_params = self.params["archi_params"][model_type]
                if model_type == ModelType.SDCC:
                    archi_params["filters"]      = hp.Int("filters",      min_value=1,  max_value=10,  step=1  )
                    archi_params["conv_layers"]  = hp.Int("conv_layers",  min_value=1,  max_value=10,  step=1  )
                    archi_params["sdcc_blocks"]  = hp.Int("sdcc_blocks",  min_value=1,  max_value=4,   step=1  )
                    archi_params["lstm_nodes"]   = hp.Int("lstm_nodes",   min_value=10, max_value=200, step=10 )
                    archi_params["lstm_layers"]  = hp.Int("lstm_layers",  min_value=1,  max_value=5,   step=1  )
                    archi_params["dense_nodes"]  = hp.Int("dense_nodes",  min_value=20, max_value=500, step=20 )
                    archi_params["dense_layers"] = hp.Int("dense_layers", min_value=1,  max_value=5,   step=1  )
                elif model_type == ModelType.BOTTLENECK:
                    ... #TODO add tuning parameters for bottleneck cnn
            case TuneableParams.LR:
                self.params["learning_rate"] = hp.Float("learning_rate", min_value=1e-4, max_value=1e-2, sampling="log")
        
        # Create model
        model = self._create_model(model_type)

        return model

    @override
    def _create_model(self, model_type: Enum) -> keras.Model:
        """Returns a Keras model, uses strategy to determine batch_size."""
        start = time.time()
        print("Started Model Creation")

        # Create instance of preprocessor (will be used for getting constants)
        preproc = PreProcessor(self.PATH)

        # Determining batch size
        batch_size = int(self.params["batch_size"])

        # Create model based on specified model type
        archi_params = self.params["archi_params"][model_type]
        match model_type:
            case ModelType.SDCC:
                model = self._create_sdcc(preproc, batch_size, archi_params)
            case ModelType.BOTTLENECK:
                model = self._create_bottleneck(preproc, batch_size, archi_params)

        # Compile model
        opt_func = self.params["optimizer"].value
        optimizer = opt_func(LRScheduler(self.params["learning_rate"]))
        metric_list = list(self.metrics.values())
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=metric_list)

        # Save plot of model
        os.chdir(f"{self.PATH}models/")
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

        # Run training
        callback_list = list(self.callbacks.values())
        model.fit(x=data.train_gen, validation_data=(data.val_gen), epochs=self.params["epochs"], 
            callbacks=callback_list)

        model.save(f"{self.PATH}models/")
    