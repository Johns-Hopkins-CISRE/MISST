#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""model.py: Trains the main Multi-Channel CNN-LSTM model"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import config
import time
import keras
import threading
import tkinter as tk
import keras.backend as K
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from preprocessor import PreProcessor
from keras.layers import (
    Conv1D,
    MaxPooling1D,
    Lambda,
    Reshape,
    LSTM,
    Dense
)

class GUI():
    """Acts as a model control panel, allowing for rapid manual hyperparameter & architecture tuning"""

    model_is_running = False
    optimizers = ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]

    def __init__(self, path):
        """Creates GUI"""
        self.path = path

        root = tk.Tk()
        root.title("JH RI Training Dashboard")

        # Frames
        main = tk.Frame(root)
        main.pack()
        f1 = tk.Frame(main)
        f1.pack(side="left", padx=(20, 10), pady=20)
        f2 = tk.Frame(main)
        f2.pack(side="left", padx=(10, 10), pady=20)
        f3 = tk.Frame(main)
        f3.pack(side="left", padx=(10, 20), pady=20)

        # Graphs
        fig = Figure(figsize = (5, 2), dpi = 120)
        self.plot1 = fig.add_subplot(111)
        self.plot1.set_title("Loss (Train + Test)")
        self.plot1.set_xlabel("Batches")
        self.plot1.set_ylabel("Loss")
        self.canvas = FigureCanvasTkAgg(fig, master = f2)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="top", expand=True, fill="both", pady=(0, 10))
        fig = Figure(figsize = (5, 2), dpi = 120)
        self.plot2= fig.add_subplot(111)
        self.plot2.set_title("Accuracy (Train + Test)")
        self.plot2.set_xlabel("Batches")
        self.plot2.set_ylabel("Accuracy (%)")
        self.canvas = FigureCanvasTkAgg(fig, master = f2)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side="bottom", expand=True, fill="both", pady=(10, 0))

        # Epoch Progress
        self.caption = ttk.Label(f1, text="Epochs", font=("Arial", 15))
        self.caption.pack(side="top")
        self.pb = ttk.Progressbar(f1, orient="horizontal", mode="determinate", length=300)
        self.pb.pack(side="top")
        self.value_label = ttk.Label(f1, text="Current Progress: 0%", font=("Arial", 11))
        self.value_label.pack(side="top")

        # Batch Progress
        self.caption2 = ttk.Label(f1, text="Batches", font=("Arial", 15))
        self.caption2.pack(side="top", pady=(20, 0))
        self.pb2 = ttk.Progressbar(f1, orient="horizontal", mode="determinate", length=300)
        self.pb2.pack(side="top")
        self.value_label2 = ttk.Label(f1, text="Current Progress: 0%", font=("Arial", 11))
        self.value_label2.pack(side="top")

        # Numerical Info
        self.iter_speed = ttk.Label(f1, text="Sec/Epoch: ", font=("Arial", 13))
        self.iter_speed.pack(side="top", pady=(20, 0), anchor="w")
        self.step_speed = ttk.Label(f1, text="Steps/Second: ", font=("Arial", 13))
        self.step_speed.pack(side="top", anchor="w")
        self.plot_time = ttk.Label(f1, text="Plot Time: ", font=("Arial", 13))
        self.plot_time.pack(side="top", anchor="w")
        
        # Start/Stop Button
        buttonborder = tk.Frame(f1,
            highlightbackground="#808080",
            highlightthickness=2,
            relief="solid")
        buttonborder.pack(side="top", pady=(20, 0))
        self.button = tk.Button(
            buttonborder, text="Start Training", command = self.toggle,
            width=20, height=5, font=("Arial", 15)
        )
        self.button.grid(column=0, row=0)

        # Model Config Label
        model_config_label = tk.Label(f3, text="Model Config", font=("Arial", 15))
        model_config_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        # User Input Fields
        epochs_label = tk.Label(f3, text="Epochs:", font=("Arial", 13))
        epochs_label.grid(row=1, column=0)
        self.epochs = tk.StringVar()
        self.epochs_i = tk.Entry(f3, textvariable=self.epochs)
        self.epochs_i.grid(row=1, column=1)

        batches_label = tk.Label(f3, text="Batches:", font=("Arial", 13))
        batches_label.grid(row=2, column=0)
        self.batches = tk.StringVar()
        self.batches_i = tk.Entry(f3, textvariable=self.batches)
        self.batches_i.grid(row=2, column=1)

        filters_label = tk.Label(f3, text="Filters:", font=("Arial", 13))
        filters_label.grid(row=3, column=0)
        self.filters = tk.StringVar()
        self.filters_i = tk.Entry(f3, textvariable=self.filters)
        self.filters_i.grid(row=3, column=1)

        lstm_label = tk.Label(f3, text="LSTM Nodes:", font=("Arial", 13))
        lstm_label.grid(row=4, column=0)
        self.lstm = tk.StringVar()
        self.lstm_i = tk.Entry(f3, textvariable=self.lstm)
        self.lstm_i.grid(row=4, column=1)

        dense_label = tk.Label(f3, text="Dense Nodes:", font=("Arial", 13))
        dense_label.grid(row=5, column=0)
        self.dense = tk.StringVar()
        self.dense_i = tk.Entry(f3, textvariable=self.dense)
        self.dense_i.grid(row=5, column=1)

        # Optimizer Multiple Choice
        optimizer_str = tk.StringVar(value=" ".join(self.optimizers))
        self.op_mc = tk.Listbox(f3, selectmode="single", exportselection=0, listvariable=optimizer_str, activestyle="none")
        self.op_mc.grid(row=6, column=0, columnspan=2, pady=20)

        # Load Default Best Inputs
        buttonborder1 = tk.Frame(f3,
            highlightbackground="#808080",
            highlightthickness=2,
            relief="solid")
        buttonborder1.grid(row=7, column=0, columnspan=2, pady=(0, 20))
        self.ld_button = tk.Button(
            buttonborder1, text="Load Defaults", command = self.load_defaults,
            width=15, height=1, font=("Arial", 15)
        )
        self.ld_button.grid(column=0, row=0)

        # Clear Inputs
        buttonborder2 = tk.Frame(f3,
            highlightbackground="#808080",
            highlightthickness=2,
            relief="solid")
        buttonborder2.grid(row=8, column=0, columnspan=2)
        self.ci_button = tk.Button(
            buttonborder2, text="Clear Inputs", command = self.clear_inputs,
            width=15, height=1, font=("Arial", 15)
        )
        self.ci_button.grid(column=0, row=0)

        root.mainloop()

    def toggle(self):
        """Toggles training: if running, stop, if not running, start"""
        if self.model_is_running:
            # Saves the model & closes the thread
            self.finished_training()
            while self.model_is_running:
                pass
        else:
            # Starts a separate thread for model training
            self.button["text"] = "Abort Training"
            self.model_is_running = True
            self.button["state"] = "disabled"
            self.ld_button["state"] = "disabled"
            self.ci_button["state"] = "disabled"
            t1 = threading.Thread(target=self.work)
            t1.start()

    def finished_training(self):
        """Resets the GUI after the model finishes training"""
        self.model_is_running = False
        self.button["text"] = "Start Training"
        self.button["state"] = "normal"
        self.ld_button["state"] = "normal"
        self.ci_button["state"] = "normal"

    def load_defaults(self):
        """Load the default parameters for training the model"""
        DEFAULT_OPTIMIZER = "adam"
        
        self.clear_inputs()
        self.epochs_i.insert(0, "100")
        self.batches_i.insert(0, "4")
        self.filters_i.insert(0, "4")
        self.lstm_i.insert(0, "50")
        self.dense_i.insert(0, "500")
        self.op_mc.selection_set(self.optimizers.index(DEFAULT_OPTIMIZER), self.optimizers.index(DEFAULT_OPTIMIZER))
    
    def clear_inputs(self):
        """Clear the inputted parameters for training the model"""
        self.epochs_i.delete(0, tk.END)
        self.batches_i.delete(0, tk.END)
        self.filters_i.delete(0, tk.END)
        self.lstm_i.delete(0, tk.END)
        self.dense_i.delete(0, tk.END)
        self.op_mc.selection_clear(0, tk.END)

    def work(self):
        """Trains model once button is clicked"""
        # Creates single dict that holds all needed GUI objects
        gui_objs = {
            "plot1": self.plot1, 
            "plot2": self.plot2,
            "canvas": self.canvas,
            "pb": self.pb, 
            "value_label": self.value_label, 
            "pb2": self.pb2, 
            "value_label2": self.value_label2, 
            "iter_speed": self.iter_speed, 
            "step_speed": self.step_speed,
            "plot_time": self.plot_time,
            "button": self.button
        }

        # Gets all model parameters from user input
        valid_params = True
        try:
            params = {
                "epochs": int(self.epochs.get()),
                "batch_size": int(self.batches.get()),
                "filters": int(self.filters.get()),
                "lstm_nodes": int(self.lstm.get()),
                "dense_nodes": int(self.lstm.get()),
                "optimizer": self.optimizers[self.op_mc.curselection()[0]]
            }
        except ValueError:
            print("WARNING: The inputted parameters were invalid, please double check them. The training will abort.")
            valid_params = False

        # Initialize the ModelTrainer and GUICallback and train the model
        if valid_params:
            trainer = ModelTrainer(self.path, params)
            callback = GUICallback(gui_objs, params)
            trainer.enable_callbacks(callback) #NOTE: We'll need to make an if statement that runs a different fit statement if callback is undefined
            model = trainer.create_model()
            data = trainer.import_data()
            trainer.train_model(model, data)
        self.finished_training()


class GUICallback(keras.callbacks.Callback):
    """Subclasses keras's callback class to allow tf .fit() to communicate with GUI"""
    
    train_loss = test_loss = train_acc = test_acc = []

    def __init__(self, gui_objs, params):
        """Passes list of GUI Objs & model params to inside of Callback to maintain encapsulation"""
        super(GUICallback, self).__init__()
        self.gui_objs = gui_objs
        self.EPOCH_PERCENT = 100.0 / float(params["epochs"])
        self.STEP_PERCENT = 100.0 / float(params["batch_size"] * params["epochs"])
        self.BATCH_SIZE = params["batch_size"]
    
    def on_train_begin(self):
        """Re-enables the stop training button"""
        self.gui_objs["button"]["state"] = "normal"

    def on_train_batch_end(self, batch, logs=None):
        """Updates for training batches"""
        self.batch_update()
        self.train_loss.append(logs["loss"])
        self.test_acc.append
    
    def on_test_batch_end(self, batch, logs=None):
        """Updates for test batches"""
        self.batch_update()
        self.test_loss.append(logs["loss"])
        self.test_acc.append(logs[logs.keys()[1]])
    
    def batch_update(self):
        """Checks for button press and updates batches progress bar"""
        self.gui_objs["pb2"]["value"] += self.STEP_PERCENT
        self.gui_objs["value_label2"]["text"] = f"Current Progress: {self.gui_objs['pb2']['value']}%"
        if self.gui_objs["button"]["text"] == "Start Training":
            self.model.stop_training = True
    
    def on_epoch_start(self, epoch, logs=None):
        """Records starting time for iteration time in seconds"""
        self.epoch_start = time.time()
    
    def on_epoch_end(self, epoch, logs=None):
        """Updates plots, progress bars, and text info"""
        # Update progres bars
        plot_s = time.time()
        self.gui_objs["pb2"]["value"] = 0
        self.gui_objs["value_label2"]["text"] = "Current Progress: 0%"
        self.gui_objs["pb"]["value"] += self.EPOCH_PERCENT
        self.gui_objs["value_label"]["text"] = f"Current Progress: {self.gui_objs['pb']['value']}%"
        
        # Update iter speed
        self.gui_objs["iter_speed"]["text"] = f"Sec/Epoch: {time.time() - self.epoch_start}s"
        
        # Update Plot 1 
        self.gui_objs["plot1"].scatter(self.train_loss, color="blue")
        self.gui_objs["plot1"].plot(self.train_loss, color="blue")
        self.gui_objs["plot1"].scatter(self.test_loss, color="red")
        self.gui_objs["plot1"].plot(self.test_loss, color="red")

        # Update Plot 2
        self.gui_objs["plot2"].scatter(self.train_acc, color="blue")
        self.gui_objs["plot2"].plot(self.train_acc, color="blue")
        self.gui_objs["plot2"].scatter(self.test_acc, color="red")
        self.gui_objs["plot2"].plot(self.test_acc, color="red")

        # Update Canvas
        self.gui_objs["canvas"].draw()

        # Update plot time
        self.gui_objs["plot_time"]["text"] = f"Plot Time: {plot_s - time.time()}s"

class ModelTrainer():
    """Creates and Trains Model"""
    
    callbacks = None

    def __init__(self, path, params):
        """Initializes class level variables, params is just user inputted values"""
        self.PATH = path
        self.params = params

    def enable_callbacks(self, callback):
        """Enables callbacks while still giving the option to use it separate of GUI"""
        self.callbacks = callback

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
        pool = []
        for _ in range(0, NUM_CHANNELS):
            inputs.append(keras.Input(batch_input_shape=(self.params["batch_size"], int(RECORDING_LEN * SAMPLE_RATE), 1)))
            conv = Conv1D(filters=self.params["filters"], kernel_size=10, activation="relu", padding="same")(inputs[-1])
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
        output = Dense(units=NUM_CLASSES)(dense2)

        # Create Model
        model = keras.Model(inputs=inputs, outputs=output)

        # Save plot of model
        keras.utils.plot_model(model, to_file="Keras_Model_Plot.png", show_shapes=True)

        elapsed = time.time() - start
        print(f"Finished creating model || Elapsed time: {elapsed}s")

        return model
    
    def import_data(self):
        print("stub")
    
    def train_model(self, model, data):
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(self.PATH + "08 Other Files/")
        model.compile(optimizer=self.params["optimizer"], metrics=["accuracy"])
        model.fit(epochs=self.params["epochs"], callbacks=[model_checkpoint_callback])
        print("stub")
    

if __name__ == "__main__":
    """Trains the model on the preprocessor.py data"""
    
    GUI(config.PATH) # Starts the GUI Class
    # trainer = ModelTrainer(config.PATH) # Starts the ModelTrainer w/o GUI
    # params = {
    # "epochs": 100
    # "batch_size": 4
    # "filters": 4
    # "lstm_nodes": 50
    # "dense_nodes": 500
    # "optimizers": "adam"
    # }
    # model = trainer.create_model()
