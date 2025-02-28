"""Module for handling datasets."""

import enum
from dataclasses import dataclass

import numpy as np
import pandas as pd


# class FingerprintEum(enum.StrEnum):
#     ATOMPAIR = enum.auto()
#     MACCS = enum.auto()
#     ECFP6 = enum.auto()
#     ECFP4 = enum.auto()
#     FCFP4 = enum.auto()
#     FCFP6 = enum.auto()
#     TOPTOR = enum.auto()
#     RDK = enum.auto()
#     AVALON = enum.auto()

FINGERPRINT_TYPES = ['ATOMPAIR', 'MACCS', 'ECFP6', 'ECFP4', 'FCFP4', 'FCFP6', 'TOPTOR', 'RDK', 'AVALON']

@dataclass
class Dataset:
    """Basic dataset class holding a dataset."""

    x_col: str
    filename: str
    y_col: str = "Label"
    X: np.ndarray = None
    y: np.ndarray = None

    def __post_init__(self):
        
        if self.x_col not in FINGERPRINT_TYPES:
            raise ValueError("Invalid fingerprint type")

        # Read data from parquet file
        df = pd.read_parquet(self.filename, columns=[self.x_col, self.y_col])

        # Process y values
        self.y = df[self.y_col].values
        # Delete y_col from df to save memory after extracting y values
        df = df.drop(columns=[self.y_col])

        # Check if y contains non-binary values
        if not np.all(np.isin(self.y, [0, 1])):
            raise ValueError("y must contain only binary labels (0 or 1)")

        first_row = np.fromstring(df[self.x_col].iloc[0], sep=",", dtype=np.float32)
        self.x = np.empty((len(df), len(first_row)), dtype=np.float32)
        for i, x in enumerate(df[self.x_col].values):
            self.x[i, :] = np.fromstring(x, sep=",", dtype=np.float32)

        # Check for NaN values
        invalid_mask = np.isnan(self.x).any(axis=1)
        invalid_rows = np.where(invalid_mask)[0]
        if len(invalid_rows) > 0:
            print(f"Warning: Found {len(invalid_rows)} invalid rows in dataset")

        # Free memory
        del df
