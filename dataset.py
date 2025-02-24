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

        if not self.test and self.y_col != 'Label':
            raise ValueError("y_col must be 'Label' for binary classification")

        df = pd.read_parquet(self.filename, columns=[self.X_col, self.y_col])
        self.y = df[self.y_col].values
        self.X = np.vstack([np.fromstring(x, sep=',') for x in df[self.X_col].values])
        if np.all(np.isin(self.y, [0, 1])):
            raise ValueError("y must contain only binary labels (0 or 1)")

        invalid_rows = np.where(pd.isna(self.X))[0]
        if len(invalid_rows) > 0:
            print(f"Warning: Found {len(invalid_rows)} invalid rows at indices: {invalid_rows.tolist()}")
