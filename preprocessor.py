#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""preprocessor.py: Imports PSG .edf & hypnogram, exports a .csv dataset"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import config
import os
import sys
import glob
import time
import mne
import pandas as pd
import numpy as np
import tensorflow as tf
from typing import Tuple
from tqdm import tqdm


class PreProcessor():
    """Peforms the actual preprocessing operations"""
    
    # Configurable Constants
    RECORDING_LEN = 10 # 10 seconds
    DOWNSAMPLING_RATE = 10 # Must be a factor of sample_rate, 10x downsample, 1000 samples per 10 secs
    RECORDINGS_PER_MOUSE = 100 # 100 10 sec recordings for every mouse
    ANNOTATIONS = { # Must be whole numbers increasing by one
        "SLEEP-S0": 0,
        "SLEEP-S2": 1,
        "SLEEP-REM": 2
    }
    DATASET_SPLIT = { # Needs to be out of 17
        "TRAIN": 13,
        "TEST": 2,
        "VAL": 2
    }
    RANDOM_SEED = 952 # Random seed used by NumPy random generator
    MARGIN = 0.01 # Margin used for determining if float is int

    # Global Vars
    mins = None
    maxs = None
    valid_dirs = None
    __sample_rate = None

    def __init__(self, path: str):
        self.PATH = path
    
    def import_example_edf(self) -> mne.io.BaseRaw:
        """Returns a single RawEDF for the purpose of determining edf info"""
        try:
            return mne.io.read_raw_edf(self.PATH + "01 Raw Data/1/BASAL FEMALE B6 280 20211019 LJK - EDF.edf")
        except FileNotFoundError:
            raise FileNotFoundError(
                "The example edf could not be found," +
                "try checking the 01 Raw Data directory structure."
            )

    def get_edf_info(self, edf: mne.io.BaseRaw) -> Tuple[float, int]:
        """For a given edf, returns the sample rate and number of channels"""
        sample_rate = edf.info["sfreq"] / self.DOWNSAMPLING_RATE
        assert abs(sample_rate - round(sample_rate)) < self.MARGIN
        num_channels = len(edf.info["ch_names"])
        return sample_rate, num_channels

    def import_and_preprocess(self, mode: str) -> None:
        """Saves edf files"""
        # Declare paths
        raw_dir = self.PATH + "01 Raw Data/"
        of_path = self.PATH + "08 Other files/"

        # Search all directories to find number of valid directories
        os.chdir(raw_dir)
        all_dirs = os.listdir()
        np.random.default_rng(seed=self.RANDOM_SEED).shuffle(all_dirs)
        valid_dirs = []
        for directory in tqdm(all_dirs, desc="Searching for Valid Directories", file=sys.stdout):
            os.chdir(raw_dir + str(directory))
            filename = glob.glob("*EDF.edf")
            if len(filename) > 0:
                valid_dirs.append(directory)
        
        # Get index of selected mode and generate start and end vals for the slice
        assert sum(self.DATASET_SPLIT.values()) == len(valid_dirs)
        mode_ind = [ind for ind, key in enumerate(self.DATASET_SPLIT) if key == mode][0]
        split_list = [self.DATASET_SPLIT[key] for key in self.DATASET_SPLIT]
        start = 0
        for ind in range(mode_ind):
            start += split_list[ind]
        end = start + self.DATASET_SPLIT[mode]

        # Set directories as a class variable
        valid_dirs = valid_dirs[start:end]

        # Traverse through sliced directories
        for directory in tqdm(valid_dirs, desc=f"Preprocessing {mode} Data", file=sys.stdout):
            # Search directory for edf
            os.chdir(raw_dir + str(directory))
            filename = glob.glob("*EDF.edf")

            if len(filename) > 0:
                # Read edf if found
                filename = filename[0]
                edf_dir = raw_dir + str(directory) + "/" + filename
                edf = mne.io.read_raw_edf(edf_dir)

                # Import annotations & remove unnecessary columns
                df = pd.read_csv("hynogram.csv")
                df = df.drop(labels=["location", "startval", "stopval", "change"], axis=1)
                df = df.sort_values("start", axis=0)
                labels = df.to_numpy()

                # Load in mins and maxs, and calculate and save if they don't already exist
                t_d = time.time()
                os.chdir(of_path)
                if self.mins is None and self.maxs is None:
                    try:
                        minmax = np.load("minmax.npz")
                        self.maxs = minmax["maxs"]
                        self.mins = minmax["mins"]
                    except OSError:
                        desc = edf.describe(data_frame=True)
                        self.maxs = desc.loc[:, "max"].to_numpy()
                        self.mins = desc.loc[:, "min"].to_numpy()
                        np.savez("minmax.npz", mins=self.mins, maxs=self.maxs)
                e_d = time.time() - t_d
                print(f"Max & Min Read Time: {e_d:.2f}s")

                # Downsample edf and assert that sample_rate is an int within margin of "MARGIN"
                t_e = time.time()
                self.__sample_rate = edf.info["sfreq"] / self.DOWNSAMPLING_RATE
                assert abs(self.__sample_rate - round(self.__sample_rate)) < self.MARGIN
                self.__sample_rate = int(self.__sample_rate)
                edf = edf.resample(sfreq=self.__sample_rate, npad='auto')
                e_e = time.time() - t_e
                print(f"Downsampling Elapsed Time: {e_e:.2f}s")

                # Find difference between start times for annotations and PSG
                edf_start = edf.info["meas_date"]
                edf_sec = edf_start.hour * 3600 + edf_start.minute * 60 + edf_start.second
                start_time = labels[0][1].split(" ")[1].split(":") # Read first data point
                start_sec = int(start_time[0]) * 3600 + int(start_time[1]) * 60 + float(start_time[2])
                start_diff = (start_sec - edf_sec) * self.__sample_rate
                # Remove data & delete unused edf variable
                edf_array = edf.get_data()[:, int(start_diff):]
                del edf
                # Calculate remaining offset
                rec_samp = self.__sample_rate * self.RECORDING_LEN 
                event_samples = int(rec_samp * len(labels))
                offset = np.size(edf_array, axis=1) - event_samples
                # Account for offset
                if offset >= 0:
                    # Remove samples from the end of the edf list to account for offset
                    edf_array = edf_array[:, :-offset]
                elif offset < 0:
                    # Remove excess labels from end of hypnogram file and edf list
                    label_offset = -1 * (offset / rec_samp)
                    add_offset = 1 - (label_offset % 1)
                    labels = labels[:-int(label_offset + add_offset) or None]
                    edf_offset = int(add_offset * rec_samp)
                    edf_array = edf_array[:, :-edf_offset or None]
                    event_samples = int(rec_samp * len(labels))

                # Don't append data if it's not a 1:1 ratio
                if event_samples != int(np.size(edf_array, axis=1)):
                    print(
                        f"WARNING: Directory \"{directory}\" " + 
                        "The number of events in events.csv did not match the number of samples in the .edf PSG. " +
                        "The .csv file likely contains more events than the .edf file, check if the .edf is corrupt."
                    )
                    continue
                
                # Remove unbalanced data
                annots, remove = self.__proc_labels(labels)
                x = self.__proc_edf(edf_array, remove)

                # Ensure balancing removal and downsampling worked correctly
                assert len(x) == len(annots)

                # Normalize data
                x_norm = []
                for sample in tqdm(x, desc="Normalize Values", file=sys.stdout):
                    sample_norm = []
                    for ch_num, channel in enumerate(tqdm(sample, leave=False, file=sys.stdout)):
                        channel_norm = []
                        for value in channel:
                            channel_norm.append((value - self.mins[ch_num]) / (self.maxs[ch_num] - self.mins[ch_num]))
                        sample_norm.append(channel_norm)
                    x_norm.append(sample_norm)
                x_norm = np.array(x_norm)

                # Create folder and save data
                t_j = time.time()
                newpath = of_path + f"{mode}/"
                if not os.path.exists(newpath):
                    os.makedirs(newpath)
                os.chdir(newpath)
                np.savez(f"{directory}.npz", x_norm=x_norm, annots=annots, allow_pickle=False)
                e_j = time.time() - t_j
                print(f"Save preprocessed data: {e_j:.2f}s")
            
            # The set_dirs function is supposed to catch all empty directories
            else:    
                raise ValueError(f"Directory \"{directory}\" Did not contain a valid .edf file")
    
    def __proc_labels(self, labels):
        """Balance labels"""
        # Remove everything besides annotations, encode, then find frequency of label
        annots = []
        samp_freq = [0] * len(self.ANNOTATIONS)
        for event in labels:
            encoded = self.ANNOTATIONS[event[0]]
            annots.append(encoded)
            samp_freq[encoded] += 1
        annots = np.array(annots)

        # Scramble annots
        t_b = time.time()
        rng = np.random.default_rng(seed=self.RANDOM_SEED)
        rng.shuffle(annots)
        e_b = time.time() - t_b
        print(f"Shuffle Annotations: {e_b:.2f}s")

        # Calculate how many samples must be removed from each category
        min_freq = np.argmin(samp_freq)
        diff = [freq - samp_freq[min_freq] for freq in samp_freq]

        # Find all unbalanced annotations
        t_i = time.time()
        remove = []
        for index, category in enumerate(annots):
            for diff_category, freq in enumerate(diff):
                if category == diff_category and freq > 0:
                    diff[diff_category] -= 1
                    remove.append(index)
        e_i = time.time() - t_i
        print(f"Search for Unbalanced Indexes: {e_i:.2f}s")
        
        # Remove unbalanced annotations
        t_c = time.time()
        annots = np.delete(annots, remove, axis=0)
        e_c = time.time() - t_c
        print(f"Annotation Removal Time: {e_c:.2f}s")
        
        # Return modified annots and remove indexes for __proc_edf()
        return annots, remove
    
    def __proc_edf(self, edf_array, remove):
        """Balance edf array; must be executed after __proc_labels()"""
        # Split data into samples
        t_f = time.time()
        prev = 0
        x = []
        rec_samp = self.RECORDING_LEN * self.__sample_rate
        for next_ in range(rec_samp, edf_array.shape[1] + rec_samp, rec_samp):
            slice_ = edf_array[:, prev:next_]
            x.append(slice_)
            prev = next_
        x = np.array(x)
        e_f = time.time() - t_f
        print(f"EDF Sampling: {e_f:.2f}s")

        # Shuffle EDF Array
        t_g = time.time()
        rng = np.random.default_rng(seed=self.RANDOM_SEED)
        rng.shuffle(x, axis=0)
        e_g = time.time() - t_g
        print(f"Shuffle EDFs: {e_g:.2f}s")
        
        # Remove unbalanced samples
        t_h = time.time()
        x = np.delete(x, remove, axis=0)
        e_h = time.time() - t_h
        print(f"EDF Removal Time: {e_h:.2f}s")

        # Return balanced array
        return x
        

if __name__ == "__main__":
    # Import, preprocess, and save the train, test, and validation sets
    preproc = PreProcessor(config.PATH)
    preproc.import_and_preprocess("TRAIN")
    preproc.import_and_preprocess("TEST")
    preproc.import_and_preprocess("VAL")
