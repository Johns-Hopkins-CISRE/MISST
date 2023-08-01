<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/Johns-Hopkins-CISRE/MISST/blob/main/docs/img/Logo%20White.png?raw=true">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/Johns-Hopkins-CISRE/MISST/blob/main/docs/img/Logo%20Black.png?raw=true">
    <img alt="MISST's Logo" width="30%" height="30%" src="https://github.com/Johns-Hopkins-CISRE/MISST/blob/main/docs/img/Logo%20Black.png?raw=true">
  </picture>
  <br>
  A Murine-based Intelligent Sleep Staging Tool
  <br>
  <img src="https://img.shields.io/badge/version-0.1.1--alpha-blue?style=for-the-badge" alt="Version Number 0.1.1-Alpha">
  <img src="https://img.shields.io/github/commit-activity/y/Johns-Hopkins-CISRE/MISST?style=for-the-badge" alt="Commit Frequency">
</h1>

MISST is a **Python library** that utilizes **deep neural networks** in order to automate the grading of **murinae polysomnograms** (PSGs). By using a combination of residual connections, bottleneck blocks, and fully-connected layers, MISST is able to classify raw PSGs with little to no preprocessing, while still remaining **lightweight** and **consistent**. On a dataset of ~10 hours of murine PSGs, MISST reached a validation accuracy of 87.6% and a Cohen's Kappa of 0.74.

This program was developed as part of a research project conducted at the *Johns Hopkins Center for Interdisciplinary Sleep Research and Education*.

# Features
- Easy-to-use Python API
- Scalable for large datasets
  - All data manipulation (preprocessing, training, etc.) uses the same amount of RAM, regardless of dataset size.
- Full GUI for Real-Time Data Analysis
- Support for both Hyperband Tuning and Bayesian Optimization
- Distributed Training (Model-Parallel & Data-Parallel)

# GUI

<img align="right" width="60%" src="https://github.com/Johns-Hopkins-CISRE/MISST/blob/main/docs/img/GUI_Full.png?raw=true">

- **Automatic** Model Configuration Generator
  - A model configuration menu is automatically generated as per the user's entered parameters
- **Live** accuracy and loss metrics
  - Updated in real-time during training
- **Prediction** Error Distribution graphs
  - Incorporates a custom metric to directly interface with the model predictions
- **Progress** bars for progress tracking

<img align="center" width="100%" height="1" src="https://github.com/Johns-Hopkins-CISRE/MISST/blob/main/docs/img/HD_transparent_picture.png">

# Getting Started
System Requirements:
- Python 3.10 or later
- A system with 8 GB RAM or more
- At least 30 GB of free disk space
- Windows, Unix, or MacOS

To install MISST, run the following command:
```shell
C:\...> pip install misst
```

# Using MISST
At a high-level, MISST can be thought of as TensorFlow-based *model training application* designed specifically for processing murine polysomnograms.

Importing MISST into a Python project is as simple as `import misst`. However, in order to *use* any of MISST's functions, there are several **supplementary files** that must be placed alongside your Python script. Conceptually, MISST uses the following **file pipeline**:

<img align="center" width="100%" src="https://github.com/Johns-Hopkins-CISRE/MISST/blob/main/docs/diagrams/file_pipelines.png?raw=true">

In practice, the **directory tree** of a MISST project should look as follows:
```shell
└──Your_Project
    ├──data
    │  ├───[subdirectory_1]
    │  ├───[subdirectory_2]
    │  └───[subdirectory_3]
    ├──my_program.py
    └──config.yaml
```
- `my_program.py` represents the script that you're writing. While it *can* contain anything, you can find a barebones example [here](https://github.com/Johns-Hopkins-CISRE/MISST/blob/main/examples/jh_example/train.py).
- For the `config.yaml` file, you should always use [this](https://github.com/Johns-Hopkins-CISRE/MISST/blob/main/examples/jh_example/config.yaml) template. You can modify and tweak parameters accordingly, but deleting an entry altogether may result in an error.
- The `data` directory must abide by strict formatting rules, which is outlined in the [Dataset](https://github.com/Johns-Hopkins-CISRE/MISST/tree/main#dataset-formatting) section of this guide.

For an example of how this should look, reference the [example](https://github.com/Johns-Hopkins-CISRE/MISST/tree/main/examples/jh_example) folder in the GitHub Repo.

# Dataset Formatting
In order to use MISST, a specific **dataset format** must be followed. This section outlines the rules for both the PSG and Hypnogram files. The program may result in an error if this formatting is not strictly adhered to.

The dataset must be placed into the `~/MISST/mist/data/` directory, with the following structure:
- Each PSG recording **must** be in it's own subdirectory
  - The subdirectory can be named anything
  - The subdirectory can include unrelated files
- Both a Hypnogram (in .csv format) and a PSG (in .edf format) recording must be present within each subdirectory
- The Hypnogram and PSG can be named anything, as long as the RegEx is able to properly filter it out

The final MISST directory tree should look as follows:
```shell
└──data
   └──raw
      ├───[subdirectory_1]
      │   └───EDF.edf
      │   └───hypnogram.csv
      ├───[subdirectory_2]
      │   └───EDF.edf
      │   └───hypnogram.csv
      ├───[subdirectory_3]
      │   └───EDF.edf
      │   └───hypnogram.csv
      │   ...
      └────[subdirectory_n]
          └───EDF.edf
          └───hypnogram.csv
```
Each Hypnogram ".csv" file should follow this format:
| type        | start                   | stop                    |
| ----------- | ----------------------- | ----------------------- |
| SLEEP-S0    | 2021-10-19 10:12:44.000 | 2021-10-19 10:12:54.000 |
| SLEEP-S2    | 2021-10-19 10:12:54.000 | 2021-10-19 10:13:04.000 |
| SLEEP-REM   | 2021-10-19 10:13:04.000 | 2021-10-19 10:13:14.000 |

**Disclaimer**: MISST is still in development. Johns Hopkins is not liable for any incorrect or misleading predictions outputted by the MISST model.

# Code Completion
The following table represents current progress on MISST:
<table>
  <thead>
    <tr>
      <th>Module</th>
      <th>Feature</th>
      <th>Description</th>
      <th>Current Status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">Trainer</td>
      <td>"PLAIN" Mode</td>
      <td>Basic model training mode</td>
      <td>✅ Finished</td>
    </tr>
    <tr>
      <td>"TUNER" Mode</td>
      <td>Uses a KerasTuner during training</td>
      <td>✅ Finished</td>
    </tr>
    <tr>
      <td>"GUI" Mode</td>
      <td>Provides a GUI for real-time training visualization</td>
      <td>🚧 In Progress</td>
    </tr>
    <tr>
      <td>"DIST" Mode</td>
      <td>Trains using a distributed computing network</td>
      <td>🚧 In Progress</td>
    </tr>
    <tr>
      <td rowspan="2">Predictor</td>
      <td>Hypnograms</td>
      <td>Create hypnograms from inputted polysomnogram</td>
      <td>❌ Not Started</td>
    </tr>
    <tr>
      <td>Azure Webapp</td>
      <td>Interactive web-app for Hypnogram prediction</td>
      <td>❌ Not Started</td>
    </tr>
  </tbody>
</table>

# Contributors
### Lead Developer: Hudson Liu &bull; 🖥️ GitHub [@hudson-liu](https://github.com/Hudson-Liu) &bull; 📧 Email hudsonliu0@gmail.com
### Mentor/Team Lead: [Luu Van Pham, M.D.](https://www.hopkinsmedicine.org/profiles/details/luu-pham)
### Team Member: Lenise Kim
### Logo Design: [delogodesign](https://www.fiverr.com/delogodesign/design-2-professional-logo-with-source-files) & [Tomona Oishi](https://github.com/TheIllusioner)

# Acknowledgements
MISST was developed by Hudson Liu over the course of an internship led by Luu Van Pham, M.D. (IT Director of the *Johns Hopkins Center for Interdisciplinary Sleep Research and Education*). Additionally, this project was made possible thanks to Lenise Kim's gold-standard murine sleep staging data. 

MISST's logo is the product of a combined effort by both [delogodesign](https://www.fiverr.com/delogodesign/design-2-professional-logo-with-source-files) and [Tomona Oishi](https://github.com/TheIllusioner).
