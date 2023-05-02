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
import sklearn
import numpy as np
import keras_tuner as kt
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from enum import Enum
from overrides import override
from typing import Any, Callable
from contextlib import redirect_stdout
from tqdm import tqdm
from keras.layers import (
    Conv1D,
    Concatenate,
    LSTM,
    Bidirectional,
    Dense,
    Add,
    ReLU,
    BatchNormalization,
    Dropout,
    MaxPooling1D,
    GlobalAveragePooling1D
)

from mist.trainer.utils.datasets import GeneratorDataset, ArrayDataset
from mist.trainer.utils.enum_vals import Splits
from mist.trainer.utils.trainers import DistributedTrainer, TunerTrainer

from mist.trainer.project_enums import TrainingModes, ModelType, TuneableParams
from mist.trainer.preprocessor import PreProcessor
from mist.trainer.gui import GenericGUI, GUICallback


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
    def __init__(self, path: str, batch_size: int, split: Splits, model_type: ModelType):
        """Initialize global vars"""
        self.PATH = path
        self.BATCH_SIZE = batch_size
        self.SPLIT = split
        self.MODEL_TYPE = model_type

        # Current index in list of all_rec
        self.cur_rec = 0 
        
        # Create list of all loaded samples
        self.x = []
        self.y = []

        # Prevent re-loading of same info
        self.num_batches = None

        # Find all files
        os.chdir(f"{self.PATH}data/shuffled/{self.SPLIT}/")
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
            self.num_batches = (num_segments // self.BATCH_SIZE) - 2
        return self.num_batches
    
    @override
    def __getitem__(self, idx):
        """Receives batch num and returns batch"""
        # Change cwd
        os.chdir(f"{self.PATH}data/shuffled/{self.SPLIT}/")

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
        slice_x, slice_y = self.x[:self.BATCH_SIZE], self.y[:self.BATCH_SIZE]
        match self.MODEL_TYPE:
            case ModelType.SDCC:
                slice_x, slice_y = self._sdcc_slices(slice_x, slice_y)
            case ModelType.BOTTLENECK:
                slice_x, slice_y = self._bn_slices(slice_x, slice_y)
            case other:
                raise ValueError("ModelType must have a defined associated DataGenerator method")

        # Dispose of old data (start will always be zero)
        self.x = self.x[self.BATCH_SIZE:]
        self.y = self.y[self.BATCH_SIZE:]

        return slice_x, slice_y

    def _sdcc_slices(self, x: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Reshapes the slices of x and y for the SDCC ModelType"""
        slice_x = np.transpose(np.array(x)[:, :, :, np.newaxis], axes=(1, 0, 2, 3))
        slice_x = [i for i in slice_x] # Convert to list
        slice_y = np.array(y)
        return slice_x, slice_y
    
    def _bn_slices(self, x: np.ndarray, y: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
        """Reshapes the slices of x and y for the Bottleneck ModelType"""
        slice_x = np.transpose(np.array(x), axes=(0, 2, 1))
        slice_y = np.array(y)
        return slice_x, slice_y

    @override
    def on_epoch_end(self):
        """Reset all class variables"""
        self.cur_rec = 0 
        self.x = []
        self.y = []


class LRTrackerCallback(tf.keras.callbacks.Callback):
    """Tracks the Learning Rate"""

    def __init__(self):
        """Initializes Learning Rate Log"""
        self.lr_logs = []

    @override
    def on_epoch_end(self, epoch, logs=None):
        """Appends decayed learning rate if present"""
        try:
            self.lr_logs.append(self.model.optimizer._decayed_lr("float32").numpy())
        except:
            self.lr_logs.append(self.model.optimizer._lr("float32").numpy())


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

        # Define Learning Rate Tracker Callback
        lr_tracker = LRTrackerCallback()

        return {
            "tensorboard_callback": tensorboard_callback,
            "lr_tracker": lr_tracker
        }
    
    @override
    def _preconfigured_metrics(self) -> dict[str, str | keras.metrics.Metric | Callable]:
        """Creates all metrics"""
        metrics = {
            "accuracy": keras.metrics.SparseCategoricalAccuracy(),
            "cohen_kappa": tfa.metrics.CohenKappa(num_classes=3, sparse_labels=True),
            #"f1_score": tfa.metrics.F1Score(num_classes=3) #TODO this might not be compatible w/ sparse labels
        }
        return metrics

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
        # Use preprocessor for getting constants
        RECORDING_LEN = preproc.RECORDING_LEN
        SAMPLE_RATE = preproc.get_edf_info(preproc.import_example_edf())
        NUM_CLASSES = len(preproc.ANNOTATIONS)

        # Input layer & first convolution
        input_layer = keras.Input(batch_input_shape=(batch_size, int(RECORDING_LEN * SAMPLE_RATE), len(preproc.CHANNELS)))
        init_conv = Conv1D(
            filters=archi_params["filter_mult"] * archi_params["scaling_factor"], 
            kernel_size=archi_params["init_kernel"], 
            activation="relu",
        )(input_layer)

        # Creates multiple nested CNN modules
        bouncy_convs = archi_params["conv_pattern"] + archi_params["conv_pattern"][::-1][1:]
        for cnn_ind in range(archi_params["cnn_blocks"]):
            pool = MaxPooling1D(pool_size=2, strides=2)(adder if cnn_ind != 0 else init_conv)
            # Creates multiple residual connections
            for block_ind in range(archi_params["bn_blocks"]):
                num_filters = archi_params["filter_mult"] * (cnn_ind + 1)
                # Create multiple Bottleneck blocks
                for group_ind, kernel_size in enumerate(bouncy_convs):
                    if block_ind == 0 and group_ind == 0:
                        normalize = BatchNormalization()(pool)
                    elif group_ind == 0:
                        normalize = BatchNormalization()(adder)
                    else:
                        normalize = BatchNormalization()(conv)
                    # Create ReLU
                    relu = ReLU()(normalize)
                    if group_ind == 0:
                        init_relu = relu
                    dropout = Dropout(0.2)(relu)
                    # Create convolution
                    if group_ind == len(bouncy_convs) - 1:
                        num_filters *= archi_params["scaling_factor"]
                    conv = Conv1D(
                        filters=num_filters,
                        kernel_size=kernel_size,
                        activation="relu",
                        padding="same"
                    )(dropout)
                # Uses projection in place of identity block if it's the first layer
                if block_ind == 0:
                    init_relu = Conv1D(
                        filters=num_filters,
                        kernel_size=bouncy_convs[-1],
                        activation="relu",
                        padding="same"
                    )(init_relu)
                adder = Add()([conv, init_relu])
        
        # Flatten
        normalize = BatchNormalization()(adder)
        relu = ReLU()(normalize)
        flat = GlobalAveragePooling1D()(relu)

        # Output
        dropout = Dropout(0.5)(flat)
        output = Dense(units=NUM_CLASSES, activation="softmax")(dropout)

        # Create Model
        model = keras.Model(inputs=input_layer, outputs=output)

        return model
            
    @override
    def _import_data(self) -> GeneratorDataset | ArrayDataset: 
        """Creates two generators, one for training and one for validation"""
        train_gen = DataGenerator(self.PATH, self.params["batch_size"], Splits.TRAIN, self.params["model_type"])
        val_gen = DataGenerator(self.PATH, self.params["batch_size"], Splits.VAL, self.params["model_type"])
        test_gen = DataGenerator(self.PATH, self.params["batch_size"], Splits.TEST, self.params["model_type"])
        return GeneratorDataset(train_gen, val_gen, test_gen)

    @override
    def _model_creator_wrapper(self, model_type: Enum, hp: kt.HyperParameters, param_to_tune: Any) -> keras.Model:
        """Sets the model parameters according to the KerasTuner 'hp' obj"""
        match param_to_tune:
            case TuneableParams.MODEL:
                archi_params = self.params["archi_params"][model_type]
                if model_type == ModelType.SDCC: #use prev params as defaults
                    archi_params["filters"]      = hp.Int("filters",      min_value=1,  max_value=10,  default=6,   step=1 )
                    archi_params["conv_layers"]  = hp.Int("conv_layers",  min_value=1,  max_value=10,  default=5,   step=1 )
                    archi_params["sdcc_blocks"]  = hp.Int("sdcc_blocks",  min_value=1,  max_value=4,   default=2,   step=1 )
                    archi_params["lstm_nodes"]   = hp.Int("lstm_nodes",   min_value=10, max_value=200, default=200, step=10)
                    archi_params["lstm_layers"]  = hp.Int("lstm_layers",  min_value=1,  max_value=5,   default=2,   step=1 )
                    archi_params["dense_nodes"]  = hp.Int("dense_nodes",  min_value=20, max_value=500, default=320, step=20)
                    archi_params["dense_layers"] = hp.Int("dense_layers", min_value=1,  max_value=5,   default=1,   step=1 )
                elif model_type == ModelType.BOTTLENECK:
                    archi_params["init_kernel"]    = hp.Int("init_kernel",   min_value=12, max_value=20, default=16, step=1)
                    archi_params["cnn_blocks"]     = hp.Int("cnn_blocks",    min_value=2,  max_value=6,  default=4,  step=1)
                    archi_params["bn_blocks"]      = hp.Int("bn_blocks",     min_value=1,  max_value=5,  default=3,  step=1)
                    archi_params["filter_mult"]    = hp.Int("filter_mult",   min_value=4,  max_value=32, default=16, step=4)
                    archi_params["scaling_factor"] = hp.Int("scaling_factor", min_value=1, max_value=8,  default=4,  step=1)
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
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(self.params["learning_rate"], self.params["decay_steps"], alpha=self.params["alpha"])
        optimizer = opt_func(lr_schedule)
        metric_list = list(self.metrics.values())
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=metric_list)

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
        history = model.fit(x=data.train_gen, validation_data=(data.val_gen), epochs=self.params["epochs"], 
            callbacks=callback_list)

        # Create the directory name
        dirname = f"{self.params['model_type'].name}, {self.params['epochs']} Epoch, {history.history['val_sparse_categorical_accuracy'][-1]:.3f}% Accuracy"

        # Create model directory
        model_dir = f"{self.PATH}models/{dirname}"
        if os.path.exists(model_dir):
            i = 1
            while os.path.exists(f"{model_dir} ({i})"): i += 1
            model_dir = f"{model_dir} ({i})"
        os.mkdir(model_dir)
        os.chdir(model_dir)

        # Save model
        model.save(f"{model_dir}/saved_model/")

        # Create specification directory
        spec_dir = f"./model_specifications/"
        if not os.path.exists(spec_dir):
            os.mkdir(spec_dir)
        os.chdir(spec_dir)

        # Save plot of model
        keras.utils.plot_model(model, to_file="Keras_Model_Plot.png", show_shapes=True)

        # Print out summary of model to txt
        with open("model_summary.txt", "w") as f:
            with redirect_stdout(f):
                model.summary()

        # Create vis folder
        vis_dir = f"../vis/"
        if not os.path.exists(vis_dir):
            os.mkdir(vis_dir)
        os.chdir("../vis/")

        # Create basic Training Progressional Metric Visualizations
        VIS_CONFIGS = ["no_lr", "lr"]
        PLOTTABLE_METRICS = ["loss", "sparse_categorical_accuracy", "cohen_kappa"]
        for vis_config in tqdm(VIS_CONFIGS):
            for metric in PLOTTABLE_METRICS:
                # Plot data
                plt.plot(history.history[metric], c="blue")
                plt.plot(history.history[f"val_{metric}"], c="red")

                # Handle config for plotting
                if vis_config == "lr":
                    lr_logs = np.interp(
                        self.callbacks["lr_tracker"].lr_logs, 
                        xp=(min(self.callbacks["lr_tracker"].lr_logs), max(self.callbacks["lr_tracker"].lr_logs)), 
                        fp=(min(history.history[metric]), max(history.history[metric]))
                    )
                    plt.plot(lr_logs , c="green")
                
                # Convert variable name into English
                metric_english = " ".join([word.capitalize() for word in metric.split('_')])

                # Specify plot details
                plt.title(f"{metric_english} Vs. Epochs")
                plt.xlabel("Epochs")
                plt.ylabel(metric_english)
                if vis_config == "no_lr":
                    plt.legend(["Train", "Validation"])
                elif vis_config == "lr":
                    plt.legend(["Train", "Validation", "Learning Rate"])
            
                # Save plot
                if not os.path.exists(f"./{vis_config}/"):
                    os.makedirs(f"./{vis_config}/")
                plt.savefig(f"./{vis_config}/{metric}_vs_epochs.png", bbox_inches="tight")
                plt.cla()
        
        # Sequentially gather test set predictions
        preds, truths = [], []
        for x, y_true in tqdm(data.test_gen, desc="Predicting on Test Set"):
            preds.extend(np.argmax(model.predict(x, verbose=0)))
            truths.extend(y_true)

        # Generate confusion matrix from test set
        LABELS = ["S0", "S2", "REM"]
        confusion_matrix = sklearn.metrics.confusion_matrix(truths, preds)
        cm_disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=LABELS)
        cm_disp.plot()
        os.chdir(model_dir)
        plt.savefig(f"./confusion_matrix.png")