import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os
import matplotlib.pyplot as plt


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),  #輸入層
            nn.ReLU(),                  #激活函數
            nn.Linear(128, output_dim)  #輸出層
        )
        
    def forward(self, state):
        return self.fc(state)