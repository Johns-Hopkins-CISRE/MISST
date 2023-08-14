#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""test_preprocessor.py: Runs basic unittests on MISST's preprocesor module"""

__author__ = "Hudson Liu"
__email__ = "hudsonliu0@gmail.com"

import os
import shutil
import unittest
import mne
import datetime
import numpy as np
import pandas as pd

import misst


class FullRunThroughTest(unittest.TestCase):
    """Tests if the preprocessor is able to fully preprocess example data."""

    def setUp(self):
        """
        Creates PreProcessor instances, a temporary "test.edf" file, 
        & a temporary "hypnogram.csv" file.
        """
        # Specifies configuration
        EPOCH_LEN = 10 # Seconds
        ANNOTATIONS = { # Must be whole numbers increasing by one
           "S0" : "SLEEP-S0",
           "S2" : "SLEEP-S2",
           "REM" : "SLEEP-REM"
        }
        DATASET_SPLIT = {
            "TRAIN":  5,
            "TEST":   1,
            "VAL":    1
        }
        BALANCE_RATIOS = { # The distribution of classes within each split, "None" = "Do not balance"
            "TRAIN": {"S0": None, "S2": None, "REM": None},
            "TEST":  {"S0": None, "S2": None, "REM": None},
            "VAL":   {"S0": 1,    "S2": 1,    "REM": 1   }
        }
        CHANNELS = ["EEG1", "EEG2", "EMGnu"] # Names of PSG channels that will be used
        self.path = f"{os.getcwd()}/tests/"
        self.preproc = misst.trainer.preprocessor.PreProcessor(
            self.path, EPOCH_LEN, ANNOTATIONS, DATASET_SPLIT,
            BALANCE_RATIOS, CHANNELS
        )

        # Create directories
        for dir_ in range(3):
            print(dir_)
            new_dir = f"{self.path}data/raw/{dir_}/"
            if not os.path.exists(new_dir):
                os.makedirs(f"{self.path}data/raw/{dir_}/")
            os.chdir(f"{self.path}data/raw/{dir_}/")

            # Creates test.edf
            NUM_EPOCHS = 5
            SAMPLE_RATE = 100 # Hz
            MEAS_TIME = datetime.datetime(
                year=2021, month=10, day=19,
                hour=10, minute=12, second=44, tzinfo=datetime.UTC
            )
            data = np.random.rand(len(CHANNELS), NUM_EPOCHS * EPOCH_LEN * SAMPLE_RATE)
            info = mne.create_info(CHANNELS , SAMPLE_RATE)
            edf = mne.io.RawArray(data, info).set_meas_date(MEAS_TIME)
            edf.export("test.edf", "edf", overwrite=True)

            # Creates test.csv
            pd.DataFrame(
                data=[
                    ["SLEEP-S0", "2021-10-19 10:12:44.000", "2021-10-19 10:12:54.000"],
                    ["SLEEP-S0", "2021-10-19 10:12:54.000", "2021-10-19 10:13:04.000"],
                    ["SLEEP-S0", "2021-10-19 10:13:04.000", "2021-10-19 10:13:14.000"],
                    ["SLEEP-S0", "2021-10-19 10:13:14.000", "2021-10-19 10:13:24.000"],
                    ["SLEEP-S0", "2021-10-19 10:13:24.000", "2021-10-19 10:13:34.000"]
                ], columns = ["type", "start", "stop"]
            ).to_csv("test.csv", index=False)

    def test_import_and_preprocess(self):
        """Tests the "import_and_preprocess()" func of the PreProcessor class"""
        # Runs the preprocessor
        try:
            self.preproc.import_and_preprocess()
        except AssertionError:
            self.fail("The \"import_and_preprocess()\" func had an internal fail (numeric computational fault).")
        except Exception:
            self.fail("The \"import_and_preprocess()\" func had an internal fail (logical fault).")

        # Ensures that the normalized dataset exists
        try:
            os.chdir(f"{self.path}data/normalized/")
        except FileNotFoundError:
            self.fail("The \"import_and_preprocess()\" func did not create the \"normalized\" directory.")

        # Ensures that train test val split dirs all exist
        SPLITS = ["TRAIN", "VAL", "TEST"]
        self.assertEqual(set(SPLITS), set(os.listdir()))

        # Ensures that the normalized dataset is not corrupted
        files = 0
        for split in SPLITS:
            os.chdir(f"{self.path}data/normalized/{split}/")
            for file in os.listdir():
                if file.endswith(".npz"):
                    files += 1
                else:
                    self.fail("The \"import_and_preprocess()\" func exported non-\".npz\" files")
        if files == 0:
            self.fail("The \"import_and_preprocess()\" func did not export any .npz files.")

        # Ensures that mins and maxes were recorded
        os.chdir(f"{self.path}data/")
        try:
            np.load("minmax.npz")
        except FileNotFoundError:
            self.fail("The \"import_and_preprocess()\" func did not export minmax.npz.")

    def tearDown(self):
        """Removes the temporary EDF file and any created files"""
        shutil.rmtree(f"{self.path}data/")


if __name__ == '__main__':
    """Runs unittests"""
    unittest.main()
