<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/docs/img/Logo%20White.png">
    <source media="(prefers-color-scheme: light)" srcset="/docs/img/Logo%20Black.png">
    <img alt="MIST's Logo" width="30%" height="30%" src="/docs/img/Logo%20Black.png">
  </picture>
  <br>
  A Murinae-based Intelligent Staging Tool
  <br>
  <img src="https://img.shields.io/badge/version-1.0.0--alpha-blue?style=for-the-badge" alt="Version Number 1.0.0-Alpha">
  <img src="https://img.shields.io/github/commit-activity/y/Johns-Hopkins-CISRE/MIST?style=for-the-badge" alt="Commit Frequency">
</h1>

MIST employs deep neural networks to **automate the grading of murinae polysomnograms** (PSGs). This program was developed as part of a research project conducted at the *Johns Hopkins Center for Interdisciplinary Sleep Research and Education*. By using a novel combination of residual connections, Bottleneck blocks, and Stacked Dilated Causal Convolutions, MIST is able to classify raw PSGs **with little to no preprocessing**, while still remaining **lightweight** and **consistent**.

Despite the plethora of existing sleep staging algorithms, few were suited to specifically grade Murinae sleep patterns. By opting to create a unique and novel network, MIST was able to achieve significantly higher accuracy than prior models. The final model reached a validation accuracy of 86.5% and a Cohen's Kappa of [insert].

# Features
- Novel Multivariate Residual SDCC/BiLSTM-based ML Architecture
- Modular PreProcessor
- All-in-one Configuration file
- Full GUI for Real-Time Data Analysis
- Support for various KerasTuners
  - Hyperband Tuner
  - Bayesian Optimization
- Distributed Training (Model-Parallel & Data-Parallel)
- Scalable for any sized dataset
  - All preprocessing has O(1) space complexity
  - Utilizes Keras Sequences for loading datasets in O(1) space complexity

# GUI

<img align="right" width="60%" src="https://github.com/Johns-Hopkins-CISRE/MIST/blob/main/docs/img/GUI_Full.png">

- **Automatic** Model Configuration Generator
  - A model configuration menu is automatically generated as per the user's entered parameters
- **Live** accuracy and loss metrics
  - Updated in real-time during training
- **Prediction** Error Distribution graphs
  - Incorporates a custom metric to directly interface with the model predictions
- **Progress** bars for progress tracking

<img align="center" width="100%" height="1" src="https://github.com/Johns-Hopkins-CISRE/MIST/blob/main/docs/img/HD_transparent_picture.png">

# Installation
Before installing MIST, make sure you meet the following prerequisites:
- Python 3.10 or later
- A system with 8 GB RAM or more
- At least 30 GB of free disk space
- Windows 10 or later (*MIST does not currently support Linux/Unix systems.*)

If you have a Windows system, run the following commands:
```shell
$ git clone https://github.com/Johns-Hopkins-CISRE/MIST.git
$ cd ~/MIST/
$ pip install requirements.txt
$ 
```

In order to use MIST on your own dataset, follow these steps:
1. Modify the PATH variable of the config.py file (located in `/MIST/mist/config.py`) to the **exact** path of your MIST installation
2. Place your dataset within the directory `~/MIST/data/raw/`
    - Each PSG recording **must** be in it's own directory
    - Multiple files can be placed within this directory, but make sure to modify the 
    - Each recording **must** be in EDF format

**Disclaimer**: MIST is still in development and has yet to pass rigorous testing. Johns Hopkins is not liable for any incorrect or misleading predictions outputted by the MIST model.

# Contributors
### Author: Hudson Liu &bull; GitHub: [@hudson-liu](https://github.com/Hudson-Liu) &bull; Email: hudsonliu0@gmail.com
### Mentor: [Luu Pham](https://www.hopkinsmedicine.org/profiles/details/luu-pham)
### Logo Design: [delogodesign](https://www.fiverr.com/delogodesign/design-2-professional-logo-with-source-files)
