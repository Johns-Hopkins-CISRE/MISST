<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="/docs/img/Logo%20White.png">
    <source media="(prefers-color-scheme: light)" srcset="/docs/img/Logo%20Black.png">
    <img alt="MIST's Logo" width="30%" height="30%" src="/docs/img/Logo%20Black.png">
  </picture>
  <br>
  A Murinae-based Intelligent Staging Tool
  <br>
  <img src="https://img.shields.io/badge/version-1.0.0--beta-blue?style=for-the-badge" alt="Version Number 1.0.0-Beta">
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
- **Statistics** for easy debugging

# Usage

# Disclaimer
MIST is still in development and has yet to pass rigorous testing.

# License
MIST is available under the MIT license. See the LICENSE file for more info.
