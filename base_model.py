#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""model.py: Trains the main Multi-Channel CNN-LSTM model"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import config
import time
import keras
import keras.backend as K
from preprocessor import PreProcessor
from keras.layers import (
    Conv1D,
    Lambda,
    Reshape,
    LSTM,
    Dense
)
    
class ModelTrainer():
    """Creates and Trains Model"""
    
    BATCH_SIZE = 4

    def __init__(self, path):
        self.PATH = path

    def create_model(self):
        """Returns a Keras model"""
        
        start = time.time()
        print("Started Model Creation")

        # Use preprocessor to get constants
        preproc = PreProcessor(self.PATH)
        RECORDING_LEN = preproc.RECORDING_LEN
        SAMPLE_RATE, NUM_CHANNELS = preproc.get_edf_info(preproc.import_example_edf())
        NUM_CLASSES = len(preproc.ANNOTATIONS)

        # Defining CNN inputs & layers
        inputs = []
        convs = []
        for _ in range(0, NUM_CHANNELS):
            inputs.append(keras.Input(batch_input_shape=(self.BATCH_SIZE, int(RECORDING_LEN * SAMPLE_RATE), 1)))
            convs.append(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs[-1]))
        
        # Vertically stacks inputs
        vstack = Lambda(lambda a: K.stack(a, axis=3))(convs)
        reshape = Reshape(target_shape=(self.BATCH_SIZE, int(RECORDING_LEN * SAMPLE_RATE), convs))
        
        # Uses Stacked-LSTM structure
        lstm1 = LSTM(units=50, return_sequences=True)(reshape)
        lstm2 = LSTM(units=50)(lstm1)

        # Dense layers
        dense1 = Dense(units=500, activation="relu")(lstm2)
        dense2 = Dense(units=500, activation="relu")(dense1)
        
        # Outputs
        output = Dense(units=NUM_CLASSES)(dense2)

        # Create Model
        model = keras.Model(inputs=inputs, outputs=output)

        # Save plot of model
        keras.utils.plot_model(model, to_file="Keras_Model_Plot.png")

        elapsed = time.time() - start
        print(f"Finished creating model || Elapsed time: {elapsed}s")

        return model
    
# This way, imports can use the class w/o running the script
if __name__ == "__main__":
    model_trainer = ModelTrainer(config.PATH)
    model = model_trainer.create_model()
