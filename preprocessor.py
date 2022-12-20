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
import pandas as pd
import numpy as np
import numpy.typing as npt
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import (
    Tuple, 
    Callable
)
from tqdm import (
    tqdm,
    trange
)


class PreProcessor():
    """Peforms the actual preprocessing operations"""
    
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
    mins = None
    maxs = None
    valid_dirs = None

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
        MARGIN = 0.01
        assert abs(sample_rate - round(sample_rate)) < MARGIN
        num_channels = len(edf.info["ch_names"])
        return sample_rate, num_channels

    def set_dirs(self, mode: str) -> Callable:
        """Sets the dataset size/range of directories that the generator searches through"""
        # Search all directories to find number of valid directories
        raw_dir = self.PATH + "01 Raw Data/"
        os.chdir(raw_dir)
        all_dirs = os.listdir()
        np.random.default_rng(seed=952).shuffle(all_dirs)
        valid_dirs = []
        for directory in tqdm(all_dirs, desc="Searching for Valid Directories"):
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
        self.valid_dirs = valid_dirs[start:end]

    def import_edf(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """Returns fully preprocessed edf data one recording at a time"""
        raw_dir = self.PATH + "01 Raw Data/"
        # Traverse through sliced directories
        for directory in tqdm(self.valid_dirs, desc="Preprocessing Data"):
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

                # Set scramble
                rng = np.random.default_rng(seed=952)
                
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

                # Load in mins and maxs, and calculate and save if they don't already exist
                t_d = time.time()
                minmax_path = self.PATH + "08 Other files/"
                os.chdir(minmax_path)
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
                sample_rate = edf.info["sfreq"] / self.DOWNSAMPLING_RATE
                MARGIN = 0.01
                assert abs(sample_rate - round(sample_rate)) < MARGIN
                sample_rate = int(sample_rate)
                edf = edf.resample(sfreq=sample_rate, npad='auto')
                e_e = time.time() - t_e
                print(f"Downsampling Elapsed Time: {e_e:.2f}s")

                # Find difference between start times for annotations and PSG
                edf_start = edf.info["meas_date"]
                edf_sec = edf_start.hour * 3600 + edf_start.minute * 60 + edf_start.second
                start_time = labels[0][1].split(" ")[1].split(":") # Read first data point
                start_sec = int(start_time[0]) * 3600 + int(start_time[1]) * 60 + float(start_time[2])
                start_diff = (start_sec - edf_sec) * sample_rate
                # Remove data & delete unused edf variable
                edf_array = edf.get_data()[:, int(start_diff):]
                del edf
                # Calculate remaining offset and remove samples from the end of the edf list to account for offset
                event_samples = int(sample_rate * self.RECORDING_LEN * len(labels))
                offset = np.size(edf_array, axis=1) - event_samples
                edf_array = edf_array[:, :-offset]

                # Don't append data if it's not a 1:1 ratio
                if int(event_samples) != int(np.size(edf_array, axis=1)):
                    print(
                        f"WARNING: Directory \"{directory}\" " + 
                        "The number of events in events.csv did not match the number of samples in the .edf PSG. " +
                        "The .csv file likely contains more events than the .edf file, check if the .edf is corrupt."
                    )
                    break
                
                # Split data into samples
                t_f = time.time()
                prev = 0
                x = []
                rec_samp = self.RECORDING_LEN * sample_rate
                for next_ in range(rec_samp, edf_array.shape[1] + rec_samp, rec_samp):
                    slice_ = edf_array[:, prev:next_]
                    x.append(slice_)
                    prev = next_
                x = np.array(x)
                e_f = time.time() - t_f
                print(f"EDF Sampling: {e_f:.2f}s")

                # Shuffle EDF Array
                t_g = time.time()
                rng.shuffle(x, axis=0)
                e_g = time.time() - t_g
                print(f"Shuffle EDFs: {e_g:.2f}s")
                
                # Split data into samples & remove unbalanced samples
                t_h = time.time()
                x = np.delete(x, remove, axis=0)
                e_h = time.time() - t_h
                print(f"EDF Removal Time: {e_h:.2f}s")

                # Normalize data
                x_norm = []
                for sample in tqdm(x, desc="Normalize Values"):
                    sample_norm = []
                    for ch_num, channel in enumerate(tqdm(sample, leave=False)):
                        channel_norm = []
                        for value in channel:
                            channel_norm.append((value - self.mins[ch_num]) / (self.maxs[ch_num] - self.mins[ch_num]))
                        sample_norm.append(channel_norm)
                    x_norm.append(sample_norm)
                x_norm = np.array(x_norm)

                # Ensure balancing removal and downsampling worked correctly
                assert len(x_norm) == len(annots)

                # Convert to tensors
                x_norm = tf.convert_to_tensor(x_norm)
                annots = tf.convert_to_tensor(annots)

                # Yield processed data
                yield x_norm, annots
            
            # The set_dirs function is supposed to catch all empty directories
            else:    
                raise ValueError(f"Directory \"{directory}\" Did not contain a valid .edf file")
        

class PreProcGenerator():
    """Uses the PreProcessor class to generate a tf.data.Dataset"""

    def __init__(self, path):
        """Initializes path variable to pass to PreProcessor class"""
        self.PATH = path
    
    def create_dataset(self) -> tf.data.Dataset:
        """Returns TensorFlow Datasets from the __import_edf() generator"""
        # Initialize PreProcessor class
        stats = PreProcessor(self.PATH)
        edf = stats.import_example_edf()
        sample_rate, num_channels = stats.get_edf_info(edf)
        
        # Use preprocessor to iteratively generate different splits
        datasets = []
        for mode in stats.DATASET_SPLIT:
            #preproc = PreProcessor(self.PATH, self.mins, self.maxs)
            preproc = PreProcessor(self.PATH)
            preproc.set_dirs(mode)
            dataset = tf.data.Dataset.from_generator(
                preproc.import_edf,
                output_signature=(
                    tf.TensorSpec(shape=(None, num_channels, int(sample_rate * stats.RECORDING_LEN)), dtype=tf.float64),
                    tf.TensorSpec(shape=(None,), dtype=tf.int32)
                )
            ).shuffle(buffer_size=1000).prefetch(buffer_size=tf.data.AUTOTUNE)
            datasets.append(dataset)
        return datasets[0], datasets[1], datasets[2]

if __name__ == "__main__":
    # Create dataset object
    generator = PreProcGenerator(config.PATH)
    train, test, val = generator.create_dataset()
    for x, y in train:
        print(x)
