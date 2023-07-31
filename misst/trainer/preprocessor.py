#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""preprocessor.py: Imports PSG .edf & hypnogram, exports 3 directories of .npz files"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import os
import sys
import shutil
import mne
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import glob
import re

from misst.trainer.utils.error_handler import short_err


class MissingChannelsException(Exception):
    """Acts as a flag for when a PSG is missing channels"""
    pass


class PreProcessor():
    """Handles preprocessing of raw polysomnogram data"""
    
    # Configurable Constants
    GROUP_LEN = 100 # Length of each segment
    RECORDING_LEN = 10 # 10 seconds
    DOWNSAMPLING_RATE = 10 # Must be a factor of sample_rate, 10x downsample, 1000 samples per 10 secs
    ANNOTATIONS = { # Must be whole numbers increasing by one
        "SLEEP-S0":  0,
        "SLEEP-S2":  1,
        "SLEEP-REM": 2
    }
    DATASET_SPLIT = {
        "TRAIN":  5,
        "TEST":   1,
        "VAL":    1
    }
    BALANCE_RATIOS = { # The distribution of classes within each split, "None" = "Do not balance"
        "TRAIN": {"S0": None, "S2": None, "REM": None}, # 2, 3, 1
        "TEST":  {"S0": None, "S2": None, "REM": None},
        "VAL":   {"S0": 1,    "S2": 1,    "REM": 1   }
    }
    CHANNELS = ["EEG1", "EEG2", "EMGnu"] # Names of PSG channels that will be used
    RANDOM_SEED = 952 # Random seed used by NumPy random generator
    MARGIN = 0.01 # Margin used for determining if float is int

    def __init__(self, path: str, edf_regex: str = None, hypnogram_regex: str = None):
        self.PATH = path
        self.EDF_REGEX = edf_regex
        self.HYPNOGRAM_REGEX = hypnogram_regex
        
        self.__mins = None
        self.__maxs = None
        self.__sample_rate = None

    def __glob_re(self, regex: str, strings: str):
        """Applies RegEx to any list of strings"""
        return list(filter(re.compile(regex).match, strings))
    
    def __parse_list(self, main_list: list[int], ratio: list[int]) -> list[int]:
        """Splits a list up into sublists of the ratio specified by parser"""
        total = sum(ratio)
        parse_lens = [round(len(main_list) * (val / total)) for val in ratio]
        parsed = []
        start = 0
        for length in parse_lens:
            parsed.append(main_list[start:start+length])
            start = length
        return parsed

    def import_example_edf(self) -> mne.io.BaseRaw:
        """Returns a single RawEDF for the purpose of determining edf info"""
        try:
            return mne.io.read_raw_edf(f"{self.PATH}data/raw/1/BASAL FEMALE B6 280 20211019 LJK - EDF.edf")
        except FileNotFoundError as err:
            msg = "The example edf could not be found, try checking the 'raw' directory's structure."
            short_err(msg, err)

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

        # Traverse through directories
        valid_files = []
        for directory in tqdm(all_dirs, desc=f"Searching Directories", file=sys.stdout):
            # Search directory for edf
            os.chdir(f"{self.PATH}data/raw/{directory}/")
            dir_files = os.listdir()
            valid_edfs = self.__glob_re(self.EDF_REGEX, dir_files)
            valid_hypnograms = self.__glob_re(self.HYPNOGRAM_REGEX, dir_files)

            # Find all valid directories
            if len(valid_edfs) == 0 or len(valid_edfs) > 1 or len(valid_hypnograms) == 0 or len(valid_hypnograms) > 1:
                if len(valid_edfs) == 0:
                    print(f"Warning: Directory \"{directory}\" did not contain a valid .edf file")
                elif len(valid_edfs) > 1:
                    print(
                        f"Warning: Directory \'{directory}\" contained multiple valid .edf files." +
                        "Make sure that the \"EDF_REGEX\" variable in the global_config.py file " +
                        "only selects one file per directory."
                    )

                if len(valid_hypnograms) == 0:
                    print(f"Warning: Directory \"{directory}\" did not contain a valid hypnogram file")
                elif len(valid_hypnograms) > 1:
                    print(
                        f"Warning: Directory \'{directory}\" contained multiple valid hypnogram files." +
                        "Make sure that the \"HYPNOGRAM_REGEX\" variable in the global_config.py file " +
                        "only selects one file per directory."
                    )
            else:
                # Save edf if valid
                valid_files.append((directory, valid_edfs[0], valid_hypnograms[0]))
        
        # Break directories up into slices
        rng = np.random.default_rng(seed=self.RANDOM_SEED)
        rng.shuffle(valid_files)
        splitted_files = self.__parse_list(valid_files, list(self.DATASET_SPLIT.values()))

        # Iterate over corresponding directories for each slice
        self.__use_dir(f"{self.PATH}data/processed/", file=False)
        for ind, files_in_split in enumerate(tqdm(splitted_files, desc="Preprocessing Splits", file=sys.stdout)):
            split_name = list(self.DATASET_SPLIT.keys())[ind]
            iter_desc = f"Preprocessing {split_name} Data"
            for directory, edf, hypnogram in tqdm(files_in_split, desc=iter_desc, leave=False, file=sys.stdout):
                balance_ratio = self.BALANCE_RATIOS[split_name]
                try:
                    x, y = self.__preproc_edf_and_hypno(directory, edf, hypnogram, balance_ratio)
                except MissingChannelsException:
                    print(
                        f"Warning: The .edf file of directory \"{directory}\" has incorrect channel names " + 
                        "(no \"Raw Score\" channel). This directory will be skipped. Note that this may cause" +
                        "the train/test/val split to become inaccurate."
                    )
                    continue
                # Create folder and save data
                newpath = f"{self.PATH}data/processed/{split_name}"
                self.__use_dir(newpath, delete=False)
                os.chdir(newpath)
                np.savez(f"{directory}.npz", x=x, y=y, allow_pickle=False)
    
    def __preproc_edf_and_hypno(self, directory: str, edf_name: str, hypnogram_name: str, balance_ratio: dict[int]):
        """Imports and preprocesses a single directory"""
        # Reads edf
        cur_dir = f"{self.PATH}data/raw/{directory}/"
        os.chdir(cur_dir)
        edf = mne.io.read_raw_edf(edf_name)

        # Import annotations & remove unnecessary columns
        df = pd.read_csv(hypnogram_name)
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
            raise MissingChannelsException()
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
            msg = (f"Directory \"{directory}\": " + 
                 "The number of events in events.csv did not match the number of samples in the .edf PSG. " +
                 "The .csv file likely contains more events than the .edf file, check if the .edf is corrupt.")
            short_err(msg, ValueError(msg))
        
        # Remove unbalanced data
        annots, remove, rand_shuf = self.__proc_labels(labels, balance_ratio)
        x = self.__proc_edf(edf_array, remove, rand_shuf)

        # Downsample edf; sample rate is updated for compatibility w/ other functions
        self.__sample_rate /= self.DOWNSAMPLING_RATE
        assert abs(self.__sample_rate - round(self.__sample_rate)) < self.MARGIN
        x = x[:, :, ::self.DOWNSAMPLING_RATE]

        # Ensure balancing removal and downsampling worked correctly
        assert len(x) == len(annots)

        # Normalize data
        x_norm = self.__normalize(x)

        return x_norm, annots
    
    def __proc_labels(self, labels, balance_ratio):
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
        min_freq = min(samp_freq)
        balance_ratio = list(balance_ratio.values())
        diff = [freq - round(min_freq * balance_ratio[ind]) if balance_ratio[ind] != None else 0 for ind, freq in enumerate(samp_freq)]

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
        # Iterate through splits
        os.chdir(self.PATH + "data/processed/")
        all_splits = os.listdir()
        for split in tqdm(all_splits, desc="Regrouping Splits", file=sys.stdout):
            # Create the corresponding regrouped dir
            self.__use_dir(f"{self.PATH}data/regrouped/{split}/", delete=True)
            # Create and save recordings
            x = []
            y = []
            counter = 0
            os.chdir(f"{self.PATH}data/processed/{split}/")
            all_files = os.listdir()
            for rec_name in tqdm(all_files, desc=f"Regrouping {split} Data", leave=False, file=sys.stdout):
                # Add to data queue
                os.chdir(f"{self.PATH}data/processed/{split}/")
                rec = np.load(rec_name)
                x.extend(rec["x"])
                y.extend(rec["y"])
                # Group and save elements of x and y
                while self.GROUP_LEN <= len(x):
                    # Slice groups from data queue
                    x_group = x[:self.GROUP_LEN]
                    y_group = y[:self.GROUP_LEN]
                    # Save groupp
                    os.chdir(f"{self.PATH}data/regrouped/{split}/")
                    np.savez(f"{counter}.npz", x=x_group, y=y_group, allow_pickle=False)
                    counter += 1
                    # Delete group from queue
                    x = x[self.GROUP_LEN:]
                    y = y[self.GROUP_LEN:]
            
            # Non-Grouped Remaining Data
            print(f"Leftover Data: {len(x)}")

    def group_shuffle(self):
        """Shuffles the data group-by-group"""
        # Declare paths & list files
        os.chdir(self.PATH + "data/regrouped/")
        all_splits = os.listdir()
        for split in tqdm(all_splits, desc="Shuffling Splits", file=sys.stdout):
            # Create the corresponding shuffled dir
            self.__use_dir(f"{self.PATH}data/shuffled/{split}/", delete=True)
            
            # Get all directories in this directory
            os.chdir(f"{self.PATH}data/regrouped/{split}/")
            all_dirs = os.listdir()

            # Create a random shuffle for data
            rand_shuf = np.arange(len(all_dirs) * self.GROUP_LEN)
            rng = np.random.default_rng(seed=self.RANDOM_SEED)
            rng.shuffle(rand_shuf)

            # Create ranges for each recording
            rec_ranges = [directory * self.GROUP_LEN for directory in range(len(all_dirs) + 1)]

            # Iterate over random indexes
            seg_x = [] # Running Segments for x
            seg_y = [] # Running Segments for y
            for counter, ind in enumerate(tqdm(rand_shuf, desc=f"Shuffling {split} Data", leave=False, file=sys.stdout)):
                if counter % (self.GROUP_LEN - 1) == 0 and counter != 0:
                    os.chdir(f"{self.PATH}data/shuffled/{split}/")
                    np.savez(f"{counter}.npz", x=seg_x, y=seg_y, allow_pickle=False)
                    seg_x.clear()
                    seg_y.clear()
                for rec in range(len(rec_ranges)):
                    if rec_ranges[rec] <= ind and ind <= rec_ranges[rec+1]:
                        os.chdir(f"{self.PATH}data/regrouped/{split}/")
                        arr = np.load(all_dirs[rec])
                        seg_x.append(arr["x"][ind - rec_ranges[rec] - 1])
                        seg_y.append(arr["y"][ind - rec_ranges[rec] - 1])

    def save_len(self) -> None:
        """Saves the number of segments in the preprocessed data"""
        # Iterate over each data split
        splits = self.DATASET_SPLIT.keys()
        split_lens = {}
        for split in splits:
            # Iterate through recordings and sums length of each one
            os.chdir(f"{self.PATH}data/regrouped/{split}/")
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
            
    def __use_dir(self, newpath: str, delete=True, file=True) -> None:
        """Creates or clears an input path"""
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        elif delete:
            files = glob.glob(newpath + "*")
            if file:
                for f in files:
                    os.remove(f)
            else:
                for f in files:
                    shutil.rmtree(f)


if __name__ == "__main__":
    """Unit-Test: Tests the PreProcessor's capabilities"""
    PATH = "C:/Users/hudso/Documents/Programming/Python/JH RI/MISST/"
    EDF_REGEX = r".*EDF\.edf$"
    HYPNOGRAM_REGEX = r"\bhynogram\.csv\b"
    preproc = PreProcessor(PATH, EDF_REGEX, HYPNOGRAM_REGEX)

    # Each method can be done asynchronously (as long as they're executed in order)
    preproc.import_and_preprocess()
    preproc.regroup()
    preproc.group_shuffle()
    preproc.save_len()
    