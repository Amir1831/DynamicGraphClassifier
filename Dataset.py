"""
Dimensions of each atlas:
       AAL - 116
       BASC - 122
       Harvard Oxford (cortical and sub-cortical) - 118
"""

import torch
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
import numpy as np
class TimeSDataset(Dataset):
    def __init__(self,data_path,label_name,label_path):
        """
        Args:
        data_path: Direct path to the atlas folder.
        label_path: Path to Behavioral-HCP.csv
        label_name: Feature Name that will use for labels. [Gender , Age , ...]
        """
        # Define Path to data & labe
        self.data_path = data_path
        self.label_path = label_path
        # Define label name
        self.label_name = label_name
        # Load Behavioral-HCP.csv
        self.BehavioralHCP = pd.read_csv(self.label_path)
        # Load TimeSeries Path
        self.TimeSeriesPaths = os.listdir(data_path)
        # Define Subject IDs
        self.TimeSeriesIDs = [int(name.split("_")[0]) for name in self.TimeSeriesPaths]

    def __len__(self):
        return len(self.TimeSeriesIDs)

    def __getitem__(self,idx):
        TimeSerie = np.loadtxt(os.path.join(self.data_path,self.TimeSeriesPaths[idx]))
        Label = self.BehavioralHCP[self.BehavioralHCP["Subject"] == self.TimeSeriesIDs[idx]][self.label_name].values
        # Map Gender to Numebr
        if self.label_name == "Gender":
            Label =  0 if Label == "M" else 1
        return torch.Tensor(TimeSerie) , torch.tensor(Label)