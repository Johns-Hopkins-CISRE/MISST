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

# Getting Started
Before installing MIST, make sure you meet the following prerequisites:
- Python 3.10 or later
- Git
- A system with 8 GB RAM or more
- At least 30 GB of free disk space
- Windows 10 or later (*MIST does not currently support Linux/Unix systems.*)

If you have a Windows system, run the following commands to install MIST:
```shell
C:\...> git clone https://github.com/Johns-Hopkins-CISRE/MIST.git
C:\...> cd ./MIST/
C:\...\MIST> pip install requirements.txt
```

# Preparing The Dataset
In order to use MIST, a specific dataset format must be followed. This section outlines the rules for both the PSG and Hypnogram files. The program may result in an error if this formatting is not strictly adhered to.

The dataset must be placed into the `~/MIST/mist/data/` directory, with the following structure:
- Each PSG recording **must** be in it's own subdirectory
  - The subdirectory can be named anything
  - The subdirectory can include unrelated files
- Both a Hypnogram (in .csv format) and a PSG (in .edf format) recording must be present within each subdirectory
- The Hypnogram and PSG can be named anything, as long as the RegEx is able to properly filter it out

The final MIST directory tree should look as follows:
```shell
    ├───MIST
    │   ├───data
    │   │   ├───raw
    │   │   │   ├───[subdirectory_1]
    │   │   │   │   └───EDF.edf
    │   │   │   │   └───hypnogram.csv
    │   │   │   ├───[subdirectory_2]
    │   │   │   │   └───EDF.edf
    │   │   │   │   └───hypnogram.csv
    │   │   │   ├───[subdirectory_3]
    │   │   │   │   └───EDF.edf
    │   │   │   │   └───hypnogram.csv
    │   │   │   │   ...
    │   │   │   ├───[subdirectory_n]
    │   │   │   │   └───EDF.edf
    │   │   │   │   └───hypnogram.csv
```
Each Hypnogram ".csv" file should follow this format:
| type        | start                   | stop                    |
| ----------- | ----------------------- | ----------------------- |
| SLEEP-S0    | 2021-10-19 10:12:44.000 | 2021-10-19 10:12:54.000 |
| SLEEP-S2    | 2021-10-19 10:12:54.000 | 2021-10-19 10:13:04.000 |
| SLEEP-REM   | 2021-10-19 10:13:04.000 | 2021-10-19 10:13:14.000 |

# Configuration
Before using MIST, the config.py file must be properly configured. To configure it, follow these steps:
1. Modify the `PATH` variable in the global_config.py file (located in `~/MIST/mist/global_config.py`) to the **exact** path of your MIST installation
2. Place your dataset within the directory `~/MIST/data/raw/`
    - Make sure your dataset follows the guidelines outlined in the [Preparing The Dataset](https://github.com/Johns-Hopkins-CISRE/MIST/tree/main#preparing-the-dataset) section
3. Since each subdirectory is allowed to have multiple EDF files, RegEx patterns are used to filter out all other undesired files. The first RegEx pattern is for the EDF file, and it can be modified in the config.py file by changing the value of `EDF_REGEX` to the desired RegEx. The second RegEx pattern is for the Hypnogram files, and can be modified by changing the value of `HYPNOGRAM_REGEX`. 
    - Both RegEx filters must filter out all but one EDF/Hypnogram

# Generating Annotations
To create annotations for your given dataset using MIST's **pretrained** model, just navigate to the directory in which you installed MIST, then enter the following command:
```shell
C:\...\MIST> python -m mist --mode generate
```
The annotations will be exported in the directory`~/MIST/data/annots/`. Each PSG recording will get it's own individual annots file, with the name of each annots file corresponding to the name of it's respective PSG recording's directory: `annots_[dir name].csv`.
This annots file will be formatted as follows:
| type      | start    | stop     |
| --------- | -------- | -------- |
| SLEEP-S0  | 10:12:44 | 10:12:54 |
| SLEEP-S2  | 10:12:54 | 10:13:04 |
| SLEEP-REM | 10:13:04 | 10:13:14 |

**Disclaimer**: MIST is still in development and has yet to pass rigorous testing. Johns Hopkins is not liable for any incorrect or misleading predictions outputted by the MIST model.
# Training
Follow these steps if you'd like to train a new model using MIST's ModelTrainer framework: 
1. Modify the `MODE` variable in the config.py file to the desired mode of operation. The following key can be used when deciding what mode of operation to use:

    | Mode Name | Description                                         | Current Status    |
    | --------- | --------------------------------------------------- | ----------------- |
    | PLAIN     | Basic model training mode                           | ✅ Working        |
    | TUNER     | Uses a KerasTuner during training                   | ✅ Working        |
    | GUI       | Provides a GUI for real-time training visualization | ❌ Not Working    |
    | DIST      | Trains using a distributed computing network        | ❌ Not Working    | 
2. Navigate to the directory in which you installed MIST, then enter the following line into the Command Prompt: 
    ```shell
    C:\...\MIST> python -m mist --mode train
    ```

# Contributors
### Lead Developer: Hudson Liu &bull; 🖥️ GitHub [@hudson-liu](https://github.com/Hudson-Liu) &bull; 📧 Email hudsonliu0@gmail.com
### Mentor/Team Lead: [Luu Van Pham, M.D.](https://www.hopkinsmedicine.org/profiles/details/luu-pham)
### Team Member: Lenise Kim
### Logo Design: [delogodesign](https://www.fiverr.com/delogodesign/design-2-professional-logo-with-source-files)

# Acknowledgements
This program was developed by Hudson Liu over the course of an internship led by Luu Van Pham, M.D. (IT Director of the *Johns Hopkins Center for Interdisciplinary Sleep Research and Education*). The pre-trained model uses data by Lenise Kim. The logo design comes from [delogodesign](https://www.fiverr.com/delogodesign/design-2-professional-logo-with-source-files).
