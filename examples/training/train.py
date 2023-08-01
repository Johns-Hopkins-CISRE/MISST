# Import necessary libraries (and MISST)
import yaml
import os
import misst

# Load the configuration file
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Get current path
path = os.getcwd() + "/"

# Run MISST's trainer
misst.preprocess_and_train(config, path)
