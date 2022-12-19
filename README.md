# Mouse PSG

This Python project uses a unique "Multivariate SDCC-BiLSTM" model to automate the process of polysomnogram grading for mice. 

# Multivariate SDCC-BiLSTM

The architecture of the network is a Multivariate SDCC-BiLSTM, which stands for "Multivariate Stacked Dilated Causal Convolutions-Bidirectional Long Short Term Memory." Essentially, this architecture is a combination of traditional LSTM approaches to time-series data, along with a multivariate implementation of WaveNet.

# Training Data

This network was trained on a variety of polysomnograms taken of mice during sleep studies at Hopkins

# Dependencies

The implementation relies primarily on Keras, TensorFlow, and MNE. 

# Distributed Computing

Distributed computing is enabled by default for training. All workers for the distributed training session must connect to the chief via LAN. 

# GUI

A GUI can also optionally be enabled, controlled by the config.MODE variable. The four available modes are listed below

Author: Hudson Liu
