import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
from functools import reduce
import operator
import json
import numpy as np
from torch.utils.data import Dataset

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    

class cost_coeffs(nn.Module):
    def __init__(self):
        super().__init__()        
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc4 = nn.Linear(32, 32)
        self.fc5 = nn.Linear(32, 21)


    def forward(self, feat):
        x = torch.relu(self.fc1(torch.cat([feat])))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return  { 'quad_cost' : x[:7], 'lin_cost' : x[7:14], 'const_cost' : x[14:] }


class M(Dataset):
    def __init__(self, X, Y):
        super().__init__()
        self.X = X
        self.Y = Y

    def __getitem__(self, i):
        return self.X[i], {"feat":self.Y["feat"][i], "pd":self.Y["pd"][i], "qd":self.Y["qd"][i],
                "pd_bus":self.Y["pd_bus"][i], "qd_bus":self.Y["qd_bus"][i]}

    def __len__(self):
        return len(self.X)
