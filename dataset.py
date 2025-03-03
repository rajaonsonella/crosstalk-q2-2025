"""Module for handling datasets."""

import enum
from dataclasses import dataclass

import numpy as np
import pandas as pd

FINGERPRINT_TYPES = ['ATOMPAIR', 'MACCS', 'ECFP6', 'ECFP4', 'FCFP4', 'FCFP6', 'TOPTOR', 'RDK', 'AVALON']

@dataclass
class Dataset:
    """Basic dataset class holding a dataset."""

    x_col: str
    filename: str
    y_col: str = "DELLabel"
    X: np.ndarray = None
    y: np.ndarray = None

    def __post_init__(self):
        
        if self.x_col not in FINGERPRINT_TYPES:
            raise ValueError("Invalid fingerprint type")

        df_info = pd.read_parquet(self.filename, columns=None)
        if self.y_col not in df_info.columns:
           df = pd.read_parquet(self.filename, columns=[self.x_col, self.y_col])
           self.y = None 
        else:
            df = pd.read_parquet(self.filename, columns=[self.x_col, self.y_col])
            self.y = df[self.y_col].values
            df = df.drop(columns=[self.y_col])
            if not np.all(np.isin(self.y, [0, 1])):
                raise ValueError("y must contain only binary labels (0 or 1)")

        first_row = np.fromstring(df[self.x_col].iloc[0], sep=",", dtype=np.float32)
        self.X = np.empty((len(df), len(first_row)), dtype=np.float32)
        for i, x in enumerate(df[self.x_col].values):
            self.X[i, :] = np.fromstring(x, sep=",", dtype=np.float32)

        invalid_mask = np.isnan(self.X).any(axis=1)
        invalid_rows = np.where(invalid_mask)[0]
        if len(invalid_rows) > 0:
            print(f"Warning: Found {len(invalid_rows)} invalid rows in dataset")

        del df
