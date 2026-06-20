import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, balanced_accuracy_score

import shap
import matplotlib.pyplot as plt

DATA_PATH  = "../misc/abide_features.csv"
SCALER_OUT = "../misc/scaler.pkl"
FEATS_OUT  = "../misc/feature_cols.pkl"
MODEL_OUT  = "../misc/brain_model.pt"
PREPARED_DATA = "../misc/abide_prepared.npz"


BATCH_SIZE = 40
EPOCHS     = 150
LR         = 1e-3
WEIGHT_DECAY = 1e-5
TEST_SIZE  = 0.4
RANDOM_STATE = 1


def load_prepared_data():

    data = np.load(PREPARED_DATA)
    feature_cols = joblib.load(FEATS_OUT)
    X_train = data['X_train']
    X_test  = data['X_test']
    y_train = data['y_train']
    y_test  = data['y_test']
    return X_train, X_test, y_train, y_test, feature_cols


class ABIDEDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
 
    def __len__(self):
        return len(self.y)
 
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class BrainMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(

            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),

            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x).squeeze(1)

 
class ModelWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.eval()
 
    def forward(self, x):
        logits = self.model(x)
        return torch.sigmoid(logits).unsqueeze(1)
 
 