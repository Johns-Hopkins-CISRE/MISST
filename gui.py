#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""gui.py: Defines all of the GUI classes: the GenericGUI abstract class and the GUICallback"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import time
import threading
import keras
import pickle
import os
import tkinter as tk
import tensorflow as tf
from tkinter import ttk
from overrides import override
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from abc import ABC, abstractmethod


class GenericGUI(ABC):
    """
    A general framework for creating a model control panel, allowing 
    for rapid manual hyperparameter & architecture tuning
    """

    OPTIMIZER = ["sgd", "rmsprop", "adam", "adadelta", "adagrad", "adamax", "nadam", "ftrl"]

    def __init__(self, path):
        """Creates GUI"""
        self.PATH = path
        self.model_is_running = False

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
        fig1 = Figure(figsize = (5, 2), dpi = 105)
        self.plot1 = fig1.add_subplot(111)
        self.plot1.set_title("Loss (Train + Test)")
        self.plot1.set_xlabel("Batch Number")
        self.plot1.set_ylabel("Loss")
        self.canvas1 = FigureCanvasTkAgg(fig1, master = f2)
        self.canvas1.draw()
        self.canvas1.get_tk_widget().pack(side="top", expand=True, fill="both", pady=(0, 10))
        fig2 = Figure(figsize = (5, 2), dpi = 105)
        self.plot2 = fig2.add_subplot(111)
        self.plot2.set_title("Accuracy (Train + Test)")
        self.plot2.set_xlabel("Batch Number")
        self.plot2.set_ylabel("Accuracy (%)")
        self.canvas2 = FigureCanvasTkAgg(fig2, master = f2)
        self.canvas2.draw()
        self.canvas2.get_tk_widget().pack(side="top", expand=True, fill="both", pady=10)
        fig3 = Figure(figsize = (5, 2), dpi = 105)
        self.plot3= fig3.add_subplot(111)
        self.plot3.set_title("Prediction Distribution (Train + Test)")
        self.plot3.set_xlabel("Class")
        self.plot3.set_ylabel("Num Predictions")
        self.canvas3 = FigureCanvasTkAgg(fig3, master = f2)
        self.canvas3.draw()
        self.canvas3.get_tk_widget().pack(side="top", expand=True, fill="both", pady=(10, 0))

        # Epoch Progress
        self.caption = ttk.Label(f1, text="Epoch 0", font=("Arial", 15))
        self.caption.pack(side="top")
        self.pb = ttk.Progressbar(f1, orient="horizontal", mode="determinate", length=300)
        self.pb.pack(side="top")
        self.value_label = ttk.Label(f1, text="Current Progress: 0%", font=("Arial", 11))
        self.value_label.pack(side="top")

        # Batch Progress
        self.caption2 = ttk.Label(f1, text="Batch 0", font=("Arial", 15))
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

        lr_label = tk.Label(f3, text="Learning Rate:", font=("Arial", 13))
        lr_label.grid(row=3, column=0)
        self.lr = tk.StringVar()
        self.lr_i = tk.Entry(f3, textvariable=self.lr)
        self.lr_i.grid(row=3, column=1)

        filters_label = tk.Label(f3, text="Filters:", font=("Arial", 13))
        filters_label.grid(row=4, column=0)
        self.filters = tk.StringVar()
        self.filters_i = tk.Entry(f3, textvariable=self.filters)
        self.filters_i.grid(row=4, column=1)

        conv_label = tk.Label(f3, text="Conv Layers:", font=("Arial", 13))
        conv_label.grid(row=5, column=0)
        self.conv = tk.StringVar()
        self.conv_i = tk.Entry(f3, textvariable=self.conv)
        self.conv_i.grid(row=5, column=1)

        sdcc_label = tk.Label(f3, text="SDCC Blocks:", font=("Arial", 13))
        sdcc_label.grid(row=6, column=0)
        self.sdcc = tk.StringVar()
        self.sdcc_i = tk.Entry(f3, textvariable=self.sdcc)
        self.sdcc_i.grid(row=6, column=1)

        lstm_label = tk.Label(f3, text="LSTM Nodes:", font=("Arial", 13))
        lstm_label.grid(row=7, column=0)
        self.lstm = tk.StringVar()
        self.lstm_i = tk.Entry(f3, textvariable=self.lstm)
        self.lstm_i.grid(row=7, column=1)

        lstm_layers_label = tk.Label(f3, text="LSTM Layers:", font=("Arial", 13))
        lstm_layers_label.grid(row=8, column=0)
        self.lstm_layers = tk.StringVar()
        self.lstm_layers_i = tk.Entry(f3, textvariable=self.lstm_layers)
        self.lstm_layers_i.grid(row=8, column=1)

        dense_label = tk.Label(f3, text="Dense Nodes:", font=("Arial", 13))
        dense_label.grid(row=9, column=0)
        self.dense = tk.StringVar()
        self.dense_i = tk.Entry(f3, textvariable=self.dense)
        self.dense_i.grid(row=9, column=1)

        dense_layers_label = tk.Label(f3, text="Dense Layers:", font=("Arial", 13))
        dense_layers_label.grid(row=10, column=0)
        self.dense_layers = tk.StringVar()
        self.dense_layers_i = tk.Entry(f3, textvariable=self.dense_layers)
        self.dense_layers_i.grid(row=10, column=1)

        # Optimizer Multiple Choice
        optimizer_str = tk.StringVar(value=" ".join(self.OPTIMIZER))
        self.op_mc = tk.Listbox(f3, selectmode="single", exportselection=0, listvariable=optimizer_str, activestyle="none")
        self.op_mc.grid(row=11, column=0, columnspan=2, pady=20)

        # Load Default Best Inputs
        buttonborder1 = tk.Frame(f3,
            highlightbackground="#808080",
            highlightthickness=2,
            relief="solid")
        buttonborder1.grid(row=12, column=0, columnspan=2, pady=(0, 20))
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
        buttonborder2.grid(row=13, column=0, columnspan=2)
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
        self.plot1.cla()
        self.plot2.clear()
        self.plot3.clear()
        self.pb["value"] = 0
        self.pb2["value"] = 0
        self.caption = "Epoch 0"
        self.caption2 = "Batch 0"
        self.value_label["text"] = "Current Progress: 0%"
        self.value_label2["text"] = "Current Progress: 0%"

    def load_defaults(self):
        """Load the default parameters for training the model"""
        DEFAULT_OPTIMIZER = "adam"
        
        self.clear_inputs()
        self.epochs_i.insert(0, "50")
        self.batches_i.insert(0, "16")
        self.lr_i.insert(0, "3.2e-4")
        self.filters_i.insert(0, "6")
        self.conv_i.insert(0, "5")
        self.sdcc_i.insert(0, "2")
        self.lstm_i.insert(0, "200")
        self.lstm_layers_i.insert(0, "2")
        self.dense_i.insert(0, "320")
        self.dense_layers_i.insert(0, "1")
        self.op_mc.selection_set(self.OPTIMIZER.index(DEFAULT_OPTIMIZER), self.OPTIMIZER.index(DEFAULT_OPTIMIZER))
    
    def clear_inputs(self):
        """Clear the inputted parameters for training the model"""
        self.epochs_i.delete(0, tk.END)
        self.batches_i.delete(0, tk.END)
        self.lr_i.delete(0, tk.END)
        self.filters_i.delete(0, tk.END)
        self.conv_i.delete(0, tk.END)
        self.sdcc_i.delete(0, tk.END)
        self.lstm_i.delete(0, tk.END)
        self.lstm_layers_i.delete(0, tk.END)
        self.dense_i.delete(0, tk.END)
        self.dense_layers_i.delete(0, tk.END)
        self.op_mc.selection_clear(0, tk.END)

    def work(self):
        """Trains model once button is clicked"""
        # Creates single dict that holds all needed GUI objects
        gui_objs = {
            "plot1": self.plot1, 
            "plot2": self.plot2,
            "plot3": self.plot3,
            "canvas1": self.canvas1,
            "canvas2": self.canvas2,
            "canvas3": self.canvas3,
            "caption": self.caption,
            "pb": self.pb, 
            "value_label": self.value_label,
            "caption2": self.caption2, 
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
                "learning_rate": float(self.lr.get()),
                "optimizer": self.OPTIMIZER[self.op_mc.curselection()[0]],
                "filters": int(self.filters.get()),
                "conv_layers": int(self.conv.get()),
                "sdcc_blocks": int(self.sdcc.get()),
                "lstm_nodes": int(self.lstm.get()),
                "lstm_layers": int(self.lstm_layers.get()),
                "dense_nodes": int(self.dense.get()),
                "dense_layers": int(self.dense_layers.get()),
            }
        except ValueError:
            print("WARNING: The inputted parameters were invalid, please double check them. The training will abort.")
            valid_params = False

        # Initialize the ModelTrainer and GUICallback and train the model
        if valid_params:
            self._train_model(gui_objs, params)
        self.finished_training()
    
    @abstractmethod
    def _train_model(self, gui_objs, params):
        """Runs whatever needs to be ran to train the model"""
        pass


class GUICallback(keras.callbacks.Callback):
    """Subclasses keras's callback class to allow tf .fit() to communicate with GUI"""

    NUM_CLASSES = 3

    def __init__(self, path, gui_objs, model_params):
        """Passes list of GUI Objs & model params to inside of Callback to maintain encapsulation"""
        super(GUICallback, self).__init__()
        self.PATH = path
        self.gui_objs = gui_objs
        self.model_params = model_params

        self.train_loss = []
        self.test_loss = []
        self.avg_test_loss = []

        self.train_acc = []
        self.test_acc = []
        self.avg_test_acc = []

        self.pred_freq = [0] * self.NUM_CLASSES
        self.true_freq = [0] * self.NUM_CLASSES
        self.y_true = None
        self.y_pred = None
        self.batch_scale = []
        self.avg_test_scale = []

        self.plot_time = 0

    @override
    def set_model(self, model):
        """Initialize variables when model is set"""
        self.model = model
        self.y_true = tf.Variable(float("nan"), dtype=model.output.dtype, shape=tf.TensorShape(None))
        self.y_pred = tf.Variable(float("nan"), dtype=model.output.dtype, shape=tf.TensorShape(None))
    
    def pred_metric(self, y_true, y_pred):
        """Fake metric that gets model predictions"""
        self.y_true.assign(y_true)
        self.y_pred.assign(y_pred)
        return 0

    @override
    def on_train_begin(self, logs=None):
        """Performs actions that signal the start of training"""
        # Re-enable the Start/Stop training button
        self.gui_objs["button"]["state"] = "normal"
        # Now that 'params' is defined, the number of epochs and number of steps can both be accessed
        self.EPOCH_PERCENT = 100.0 / float(self.params["epochs"])
        self.STEP_PERCENT = 100.0 / float(self.params["steps"])
        # Since this information isn't communicated to the callback, the program reads from 
        # the file 'split_lens.pkl' to find the length of 'validation_data'
        os.chdir(f"{self.PATH}08 Other files/")
        with open("split_lens.pkl", "rb") as f:
            split_lens = pickle.load(f)
        self.validation_length = (split_lens["VAL"] // self.model_params["batch_size"]) - 1

    @override
    def on_train_end(self, logs=None):
        """Delete the y_true and y_pred to clear up memory"""
        del self.y_true, self.y_pred

    @override
    def on_train_batch_end(self, batch, logs=None):
        """Updates for training batches"""
        # Updates general batch
        self.batch_update()
        # Update batch loss & accuracy
        self.train_loss.append(logs["loss"])
        self.train_acc.append(logs["accuracy"])
        # Update progress bars
        self.gui_objs["caption2"]["text"] = f"Batch {batch}"
        self.gui_objs["pb2"]["value"] += self.STEP_PERCENT
        self.gui_objs["value_label2"]["text"] = f"Current Progress: {self.gui_objs['pb2']['value']:.2f}%"
    
    @override
    def on_test_batch_end(self, batch, logs=None):
        """
        Updates test loss by scaling it proportionally with the training data, such that
        the end of the validation loss will align with the end of the training loss. 
        """
        # Updates general batch
        self.batch_update()
        # Update batch loss & accuracy
        self.test_loss.append(logs["loss"])
        self.test_acc.append(logs["accuracy"])
        # Update scaler for test loss & accuracy
        prev_val = self.batch_scale[-1] if len(self.batch_scale) > 0 else 0.0
        self.batch_scale.append(prev_val + (self.params["steps"] / self.validation_length))
    
    def process_model_output(self, y):
        """Takes the model output and returns which class was outputted"""
        print("stub")
        print(f"y: {y.numpy()}")
        print(f"y shape: {y.numpy().shape}")
        #I cant finish this currently w/o the actual values of y or y_shape so I'll just leave this as a stub until i can train the model

    def batch_update(self):
        """Performs all updating actions done for both training and testing batches"""
        # Start tracking time
        plot_s = time.time()
        
        # Checks for aborting button press
        if self.gui_objs["button"]["text"] == "Start Training":
            self.model.stop_training = True
        #self.process_model_output(self.y_pred) 
        self.gui_objs["plot3"].bar(["S0", "S2", "REM"], self.pred_freq)

        # Update Plot 1 
        self.gui_objs["plot1"].plot(self.train_loss, color="red")
        self.gui_objs["plot1"].plot(self.batch_scale, self.test_loss, color="blue")
        self.gui_objs["plot1"].plot(self.avg_test_scale, self.avg_test_loss, color="blue", linestyle="dashed")

        # Update Plot 2
        self.gui_objs["plot2"].plot(self.train_acc, color="red")
        self.gui_objs["plot2"].plot(self.batch_scale, self.test_acc, color="blue")
        self.gui_objs["plot2"].plot(self.avg_test_scale, self.avg_test_acc, color="blue", linestyle="dashed")

        # Update Canvas
        self.gui_objs["canvas1"].draw()
        self.gui_objs["canvas2"].draw()
        self.gui_objs["canvas3"].draw()

        # Update plot time
        self.gui_objs["plot_time"]["text"] = f"Plot Time: {(time.time() - plot_s):.2f}s"

    @override
    def on_epoch_begin(self, epoch, logs=None):
        """Records starting time for iteration time in seconds"""
        self.epoch_start = time.time()
    
    @override
    def on_epoch_end(self, epoch, logs=None):
        """Updates plots, progress bars, and text info"""

        # Update epoch accuracy
        self.avg_test_acc.append(logs["val_accuracy"])
        self.avg_test_loss.append(logs["val_loss"])
        prev_val = self.avg_test_scale[-1] if len(self.avg_test_scale) > 0 else 0
        self.avg_test_scale.append(prev_val + len(self.train_loss) - 1)

        # Update progress bars
        self.gui_objs["caption"]["text"] = f"Epoch {epoch}"
        self.gui_objs["pb"]["value"] += self.EPOCH_PERCENT
        self.gui_objs["value_label"]["text"] = f"Current Progress: {self.gui_objs['pb']['value']:.2f}%"
        self.gui_objs["caption2"]["text"] = "Batch 0"
        self.gui_objs["pb2"]["value"] = 0
        self.gui_objs["value_label2"]["text"] = f"Current Progress: 0%"

        # Update iter speed
        self.gui_objs["iter_speed"]["text"] = f"Sec/Epoch: {time.time() - self.epoch_start}s" # move update to batch, then sum delays

        # Clear out histogram values
        self.pred_freq = [0] * self.NUM_CLASSES
