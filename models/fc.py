import torch
import torch.nn as nn
import torch.nn.functional as F

class FC(nn.Module):
    def __init__(self, num_features):
        super(FC, self).__init__() 
        self.layers = nn.Sequential(
            nn.Linear(3 * 32 * 32, num_features),
            nn.ReLU(),
            nn.Linear(num_features, 10)
        )
    def forward(self, x: torch.Tensor):
        return self.layers(x.flatten(start_dim=1))
