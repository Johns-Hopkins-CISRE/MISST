# Mouse PSG

This Python project uses Multivariate CNN-LSTM networks to automate the process of polysomnogram grading for mice. 

# Training Data

This network was trained on a variety of polysomnograms taken of mice during sleep studies at Hopkins.

# Architecture

The architecture of the network is a Multivariate CNN LSTM, with a modified WaveNet being applied to each input channel.

# Dependencies

The implementation relies primarily on Keras, TensorFlow, and MNE. 

# Distributed Computing

Distributed computing is enabled by default for training. All workers for the distributed training session must connect to the chief via LAN. 

# GUI

A GUI can also optionally be enabled, controlled by the config.MODE variable. The four available modes are listed below

Author: Hudson Liu
