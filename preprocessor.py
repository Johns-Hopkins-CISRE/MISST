#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""preprocessor.py: Imports PSG .edf & hypnogram, exports a .csv dataset"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import config
import os
import glob
import time
import mne
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import (
    tqdm,
    trange
)

class PreProcessor():
    """Preprocesses the PSGs"""
    
    RECORDING_LEN = 10 # 10 seconds
    ANNOTATIONS = {
        "SLEEP-S0": 0,
        "SLEEP-S2": 1,
        "SLEEP-REM": 2
    }

    def __init__(self, path):
        self.PATH = path
    
    def import_example_edf(self):
        """Returns a single RawEDF for the purpose of determining edf info"""
        try:
            return mne.io.read_raw_edf(self.PATH + "01 Raw Data/1/BASAL FEMALE B6 280 20211019 LJK - EDF.edf")
        except FileNotFoundError:
            raise FileNotFoundError(
                "The example edf could not be found," +
                "try checking the 01 Raw Data directory structure."
            )

    def get_edf_info(self, edf):
        """For a given edf, returns the sample rate and number of channels"""
        sample_rate = edf.info["sfreq"]
        num_channels = len(edf.info["ch_names"])
        return sample_rate, num_channels

    def import_data(self):
        """Returns a list of all edf files"""
        start = time.time()
        print("Started Importing Data")

        # Find all directories & append data of shape (n_events, n_channels, n_instances) to proc_data
        raw_data_dir = self.PATH + "01 Raw Data/"
        os.chdir(raw_data_dir)
        all_dirs = os.listdir()
        proc_data = []
        for directory in tqdm(all_dirs):
            # Search directory for edf
            os.chdir(raw_data_dir + str(directory))
            edf_filename = glob.glob("*EDF.edf")

            if len(edf_filename) > 0:
                # Read edf if found
                edf_filename = edf_filename[0]
                edf_dir = raw_data_dir + str(directory) + "/" + edf_filename
                edf = mne.io.read_raw_edf(edf_dir)
                
                # Import annotations & remove unnecessary columns
                df = pd.read_csv("hynogram.csv")
                df = df.drop(labels=["location", "startval", "stopval", "change"], axis=1)
                df = df.sort_values("start", axis=0)
                labels = df.to_numpy()
                
                # Remove everything besides annotations and then encode
                annots = [self.ANNOTATIONS[event[0]] for event in labels]

                # Find the number of samples there are in the hypnogram
                sample_rate = edf.info["sfreq"]
                event_samples = int(sample_rate * self.RECORDING_LEN * len(labels))

                # Find difference between start times for annotations and PSG
                edf_start = edf.info["meas_date"]
                edf_sec = edf_start.hour * 3600 + edf_start.minute * 60 + edf_start.second
                start_time = labels[0][1].split(" ")[1].split(":") # Read first data point
                start_sec = int(start_time[0]) * 3600 + int(start_time[1]) * 60 + float(start_time[2])
                start_diff = (start_sec - edf_sec) * sample_rate
                # Remove data
                edf_array = edf.get_data()[:, int(start_diff):]
                # Calculate remaining offset and remove samples from the end of the edf list to account for offset
                remaining_offset = np.size(edf_array, axis=1) - event_samples
                edf_array = edf_array[:, :-remaining_offset]

                # Don't append data if it's not a 1:1 ratio
                if int(event_samples) != int(np.size(edf_array, axis=1)):
                    print(
                        f"WARNING: Directory \"{directory}\" " + 
                        "The number of events in events.csv did not match the number of samples in the .edf PSG. " +
                        "The .csv file likely contains more events than the .edf file, check if the .edf is corrupt."
                    )
                else:
                    # Split data into epochs and merge edf with annotations
                    prev = 0
                    proc = []
                    for annot in annots:
                        next_ = int(prev + self.RECORDING_LEN * sample_rate)
                        slice_ = edf_array[prev:next_]
                        proc.append([slice_, annot])
                        prev = next_

                    # Append processed data to proc_data
                    proc_data.extend(proc)
            else:
                print(f"WARNING: Directory \"{directory}\" Did not contain a valid .edf file")
        
        elapsed = time.time() - start
        print(f"Finished importing data || Elapsed time: {elapsed}s")

        return proc_data
    
    def preprocess(self, raw_data):
        """Inputs raw data and outputs preprocessed data"""
        print("stub")

    
if __name__ == "__main__":
    # Instantiate preprocessor obj & import data
    preprocessor = PreProcessor(config.PATH)
    proc_data = preprocessor.import_data()
    
    # Save data & time how long it takes
    start = time.time()
    joblib.dump(proc_data, config.PATH + "01 Raw Data/processed_data.joblib")
    elapsed = time.time() - start
    print(f"Finished saving data || Elapsed time: {elapsed}s")
