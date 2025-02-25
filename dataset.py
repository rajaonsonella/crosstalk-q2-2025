import pandas as pd
import numpy as np
from dataclasses import dataclass

FINGERPRINT_TYPES = ['ATOMPAIR', 'MACCS', 'ECFP6', 'ECFP4', 'FCFP4', 'FCFP6', 'TOPTOR', 'RDK', 'AVALON']

@dataclass
class Dataset:
    X_col: str
    filename: str
    y_col: str = 'Label'
    X: np.ndarray = None
    y: np.ndarray = None

    def __post_init__(self):
        if not isinstance(self.X_col, str) or self.X_col not in FINGERPRINT_TYPES:
            raise ValueError(f"X_col must be one of {FINGERPRINT_TYPES}")

        # Read data from parquet file
        df = pd.read_parquet(self.filename, columns=[self.X_col, self.y_col])
        
        # Process y values
        self.y = df[self.y_col].values
        # Delete y_col from df to save memory after extracting y values
        df = df.drop(columns=[self.y_col])
        
        # Check if y contains non-binary values
        if not np.all(np.isin(self.y, [0, 1])):
            raise ValueError("y must contain only binary labels (0 or 1)")

        first_row = np.fromstring(df[self.X_col].iloc[0], sep=',', dtype=np.float32)
        self.X = np.empty((len(df), len(first_row)), dtype=np.float32)
        for i, x in enumerate(df[self.X_col].values):
            self.X[i, :] = np.fromstring(x, sep=',', dtype=np.float32)
        
        # Check for NaN values
        invalid_mask = np.isnan(self.X).any(axis=1)
        invalid_rows = np.where(invalid_mask)[0]
        if len(invalid_rows) > 0:
            print(f"Warning: Found {len(invalid_rows)} invalid rows in dataset")
        
        # Free memory
        del df