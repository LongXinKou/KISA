import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class mlpAnnotator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(mlpAnnotator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(self.input_dim, 256) # attended feature + frame feature
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)