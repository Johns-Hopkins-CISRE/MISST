#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""preprocessor.py: Imports PSG .edf & hypnogram, exports 3 directories of .npz files"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import os
import sys
import shutil
import glob
import random
import mne
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

import config


class PreProcessor():
    """Handles preprocessing of raw polysomnogram data"""
    
    # Configurable Constants
    RECORDING_LEN = 10 # 10 seconds
    DOWNSAMPLING_RATE = 10 # Must be a factor of sample_rate, 10x downsample, 1000 samples per 10 secs
    ANNOTATIONS = { # Must be whole numbers increasing by one
        "SLEEP-S0":  0,
        "SLEEP-S2":  1,
        "SLEEP-REM": 2
    }
    DATASET_SPLIT = { # Needs to be out of 17
        "TRAIN": 13,
        "TEST":   2,
        "VAL":    2
    }
    BALANCE_RATIOS = { # The distribution of classes within each split
        "TRAIN": [1, 2, 1],
        "TEST":  [1, 1, 1],
        "VAL":   [1, 1, 1]
    }
    CHANNELS = ["EEG1", "EEG2", "EMGnu"] # Names of PSG channels that will be used
    RANDOM_SEED = 952 # Random seed used by NumPy random generator
    MARGIN = 0.01 # Margin used for determining if float is int

    def __init__(self, path: str):
        self.PATH = path
        
        self.__mins = None
        self.__maxs = None
        self.__sample_rate = None
    
    def import_example_edf(self) -> mne.io.BaseRaw:
        """Returns a single RawEDF for the purpose of determining edf info"""
        try:
            return mne.io.read_raw_edf(f"{self.PATH}data/raw/1/BASAL FEMALE B6 280 20211019 LJK - EDF.edf")
        except FileNotFoundError:
            raise FileNotFoundError(
                "The example edf could not be found, try checking the 'raw' directory's structure."
            )

    def get_edf_info(self, edf: mne.io.BaseRaw) -> float:
        """For a given edf, returns the sample rate and number of channels"""
        sample_rate = edf.info["sfreq"] / self.DOWNSAMPLING_RATE
        assert abs(sample_rate - round(sample_rate)) < self.MARGIN
        return sample_rate

    def import_and_preprocess(self) -> None:
        """Imports and preprocesses all the training data, then saves them as .npz files"""
        # Get all dirs
        os.chdir(f"{self.PATH}data/raw/")
        all_dirs = os.listdir()

        # Keep MNE from outputting unnecessary log data
        mne.set_log_level("error")

        # Traverse through sliced directories
        for directory in tqdm(all_dirs, desc=f"Preprocessing All Data", file=sys.stdout):
            # Search directory for edf
            os.chdir(f"{self.PATH}data/raw/{directory}/")
            filename = glob.glob("*EDF.edf")

            if len(filename) > 0:
                # Read edf if found
                filename = filename[0]
                edf_dir = f"{self.PATH}data/raw/{directory}/{filename}/"
                edf = mne.io.read_raw_edf(edf_dir)

                # Import annotations & remove unnecessary columns
                df = pd.read_csv("hynogram.csv")
                df = df.drop(labels=["location", "startval", "stopval", "change"], axis=1)
                df = df.sort_values("start", axis=0)
                labels = df.to_numpy()

                # Load in mins and maxs, and calculate and save if they don't already exist
                os.chdir(f"{self.PATH}data/")
                if self.__mins is None and self.__maxs is None:
                    try:
                        minmax = np.load("minmax.npz")
                        self.__maxs = minmax["maxs"]
                        self.__mins = minmax["mins"]
                    except OSError:
                        desc = edf.describe(data_frame=True)
                        self.__maxs = desc.loc[:, "max"].to_numpy()
                        self.__mins = desc.loc[:, "min"].to_numpy()
                        np.savez("minmax.npz", mins=self.__mins, maxs=self.__maxs)

                # Temporary sample rate
                self.__sample_rate = edf.info["sfreq"]

                # Find difference between start times for annotations and PSG
                edf_start = edf.info["meas_date"]
                edf_sec = edf_start.hour * 3600 + edf_start.minute * 60 + edf_start.second
                start_time = labels[0][1].split(" ")[1].split(":") # Read first data point
                start_sec = int(start_time[0]) * 3600 + int(start_time[1]) * 60 + float(start_time[2])
                start_diff = (start_sec - edf_sec) * self.__sample_rate
                
                # Ensure dates of EDF & Hypnogram agree
                edf_y, edf_m, edf_d = edf_start.year, edf_start.month, edf_start.day 
                hyp_y, hyp_m, hyp_d = [int(val) for val in labels[0][1].split(" ")[0].split("-")]
                assert (edf_y == hyp_y) and (edf_m == hyp_m) and (edf_d == hyp_d)

                # Remove data & convert to numpy
                edf_array = edf.get_data()[:, int(start_diff):]

                # Delete unused channels & unused edf obj
                try:
                    take = [edf.ch_names.index(ch) for ch in self.CHANNELS]
                    edf_array = edf_array[take]
                except ValueError:
                    print(
                        f"Warning: The .edf file of directory \"{directory}\" has incorrect channel names " + 
                        "(no \"Raw Score\" channel). This directory will be skipped."
                    )
                    continue
                del edf

                # Calculate remaining offset
                rec_samp = self.__sample_rate * self.RECORDING_LEN
                event_samples = int(rec_samp * len(labels))
                offset = np.size(edf_array, axis=1) - event_samples

                # Align the ends of "labels" and "edf_array"
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
                    raise ValueError(
                        f"Directory \"{directory}\": " + 
                        "The number of events in events.csv did not match the number of samples in the .edf PSG. " +
                        "The .csv file likely contains more events than the .edf file, check if the .edf is corrupt."
                    )
                
                # Remove unbalanced data
                annots, remove, rand_shuf = self.__proc_labels(labels)
                x = self.__proc_edf(edf_array, remove, rand_shuf)

                # Downsample edf; sample rate is updated for compatibility w/ other functions
                self.__sample_rate /= self.DOWNSAMPLING_RATE
                assert abs(self.__sample_rate - round(self.__sample_rate)) < self.MARGIN
                x = x[:, :, ::self.DOWNSAMPLING_RATE]

                # Ensure balancing removal and downsampling worked correctly
                assert len(x) == len(annots)

                # Normalize data
                x_norm = self.__normalize(x)

                # Create folder and save data
                newpath = f"{self.PATH}data/processed/"
                self.__use_dir(newpath, delete=False)
                os.chdir(newpath)
                np.savez(f"{directory}.npz", x_norm=x_norm, annots=annots, allow_pickle=False)
            # Just move onto the next directory
            else:    
                print(f"Warning: Directory \"{directory}\" did not contain a valid .edf file")
    
    def __proc_labels(self, labels):
        """Shuffle labels and removed unbalanced classes"""
        # Remove everything besides annotations, encode, then find frequency of label
        annots = []
        samp_freq = [0] * len(self.ANNOTATIONS)
        for event in labels:
            encoded = self.ANNOTATIONS[event[0]]
            annots.append(encoded)
            samp_freq[encoded] += 1
        annots = np.array(annots)

        # Scramble annots
        rng = np.random.default_rng(seed=self.RANDOM_SEED)
        rand_shuf = np.arange(len(labels))
        rng.shuffle(rand_shuf)
        annots = annots[rand_shuf]

        # Calculate how many samples must be removed from each category
        min_freq = np.argmin(samp_freq)
        diff = [freq - samp_freq[min_freq] for freq in samp_freq]

        # Find all unbalanced annotations
        remove = []
        for index, category in enumerate(annots):
            for diff_category, freq in enumerate(diff):
                if category == diff_category and freq > 0:
                    diff[diff_category] -= 1
                    remove.append(index)
        
        # Remove unbalanced annotations
        annots = np.delete(annots, remove, axis=0)
        
        # Return modified annots and remove indexes for __proc_edf()
        return annots, remove, rand_shuf
    
    def __proc_edf(self, edf_array, remove, rand_shuf):
        """Shuffle edf arrays and remove unbalanced classes; must be executed after __proc_labels()"""
        # Split data into samples
        prev = 0
        x = []
        rec_samp = int(self.RECORDING_LEN * self.__sample_rate)
        for next_ in range(rec_samp, edf_array.shape[1] + rec_samp, rec_samp):
            slice_ = edf_array[:, prev:next_]
            x.append(slice_)
            prev = next_
        x = np.array(x)

        # Shuffle .edf array
        x = x[rand_shuf]
        
        # Remove unbalanced samples
        x = np.delete(x, remove, axis=0)

        # Return balanced array
        return x

    def __normalize(self, x):
        """Normalizes data"""
        x_norm = []
        for sample in tqdm(x, desc="Normalize Values", leave=False, file=sys.stdout):
            sample_norm = []
            for ch_num, channel in enumerate(sample):
                channel_norm = []
                for value in channel:
                    channel_norm.append((value - self.__mins[ch_num]) / (self.__maxs[ch_num] - self.__mins[ch_num]))
                sample_norm.append(channel_norm)
            x_norm.append(sample_norm)
        return np.array(x_norm)
        
    def regroup(self):
        """Regroups the preprocessed data"""
        # Number of recordings per group (remainding recordings are cropped out)
        GROUP_LEN = 100

        # Create new folder
        newpath = self.PATH + "data/regrouped/"
        self.__use_dir(newpath)

        # Create and save recordings
        os.chdir(self.PATH + "data/processed/")
        all_files = os.listdir()
        x = []
        y = []
        counter = 0
        for rec_name in tqdm(all_files, desc="Regrouping Recordings"):
            # Add to data queue
            os.chdir(self.PATH + "data/processed/")
            rec = np.load(rec_name)
            x.extend(rec["x_norm"])
            y.extend(rec["annots"])
            # Group and save elements of x and y
            while GROUP_LEN <= len(x):
                # Slice groups from data queue
                x_group = x[:GROUP_LEN]
                y_group = y[:GROUP_LEN]
                # Save groups
                os.chdir(self.PATH + "data/regrouped/")
                np.savez(f"{counter}.npz", x=x_group, y=y_group, allow_pickle=False)
                counter += 1
                # Delete group from queue
                x = x[GROUP_LEN:]
                y = y[GROUP_LEN:]
        
        # Non-Grouped Remaining Data
        print(f"Leftover Data: {len(x)}")

    def group_shuffle(self):
        """Shuffles the data group-by-group"""
        # Declare paths & list files
        src = self.PATH + "data/regrouped/"
        dest = self.PATH + "data/shuffled/"
        self.__use_dir(dest)
        os.chdir(src)
        num_passes = len(os.listdir()) - 1

        # Defines groups: a & b are groups
        for _ in tqdm(range(num_passes), desc="Shuffle Pass", file=sys.stdout):
            # Set directory
            os.chdir(src)
            all_files = os.listdir()
            all_files.sort() # Ensure same order every iteration
            a_load = np.load(src + all_files[0])
            a_x, a_y = a_load["x"], a_load["y"]
            for group_num, group in enumerate(all_files[1:]):
                # Define new groups
                b_load = np.load(src + group)
                b_x, b_y = b_load["x"], b_load["y"]
                # Merge
                merged_x, merged_y = np.concatenate((a_x, b_x), axis=0), np.concatenate((a_y, b_y), axis=0)
                # Shuffle
                assert len(merged_x) == len(merged_y)
                rng = np.random.default_rng(seed=self.RANDOM_SEED + group_num)
                rand_shuf = np.arange(len(merged_x))
                rng.shuffle(rand_shuf)
                merged_x = merged_x[rand_shuf]
                merged_y = merged_y[rand_shuf]
                # Split
                full_x, full_y = np.split(merged_x, 2), np.split(merged_y, 2)
                half_x, half_y = full_x[0], full_y[0]
                a_x, a_y = full_x[1], full_y[1]
                # Save
                os.chdir(dest)
                np.savez(f"{group_num}.npz", x=half_x, y=half_y)
                # If at end of list, save "a" too
                if group_num == len(all_files[1:]) - 1:
                    np.savez(f"{group_num + 1}.npz", x=a_x, y=a_y)
            # Use previous output as current input
            src = dest

    def split_dataset(self):
        """Splits the dataset into a train, test, and val set"""
        # Declare paths
        src = self.PATH + "data/shuffled/"
        dest = self.PATH + "data/split/"

        # Create random list of all files to split
        os.chdir(src)
        all_files = os.listdir()
        random.seed(self.RANDOM_SEED)
        random.shuffle(all_files)

        # Create new folder
        modes = list(self.DATASET_SPLIT.keys())
        for mode in modes:
            self.__use_dir(dest + mode + "/")
        
        # Split files as according to dataset_split
        start = 0
        for mode_name, val in tqdm(self.DATASET_SPLIT.items(), desc="Split Dataset"):
            ratio = val / sum(self.DATASET_SPLIT.values())
            group_len = int(len(all_files) * ratio) # Does floor so that it'll never run dry of indexes
            file_slice = all_files[start:start + group_len]
            start += group_len
            for file in file_slice:
                shutil.copy(f"{src}{file}", f"{dest}{mode_name}/{file}")
    
    def save_len(self) -> None:
        """Saves the number of segments in the preprocessed data"""
        # Iterate over each data split
        splits = self.DATASET_SPLIT.keys()
        split_lens = {}
        for split in splits:
            # Iterate through recordings and sums length of each one
            os.chdir(f"{self.PATH}data/split/{split}/")
            all_recs = os.listdir()
            total_len = 0
            for filename in all_recs:
                new_rec = np.load(filename)
                x = new_rec["y"].shape[0]
                total_len += x
            split_lens.update({split: total_len})
        
        # Save data
        os.chdir(f"{self.PATH}data/")
        with open("split_lens.pkl", "wb") as f:
            pickle.dump(split_lens, f, protocol=pickle.HIGHEST_PROTOCOL)
            
    def __use_dir(self, newpath: str, delete=True) -> None:
        """Creates or clears an input path"""
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        elif delete:
            files = glob.glob(newpath + "*")
            for f in files:
                os.remove(f)


if __name__ == "__main__":
    # Import, preprocess, and save the train, test, and validation sets
    preproc = PreProcessor(config.PATH)

    # Each method can be done asynchronously (as long as they're executed in order)
    preproc.import_and_preprocess()
    preproc.regroup()
    preproc.group_shuffle()
    preproc.split_dataset()
    preproc.save_len()
    