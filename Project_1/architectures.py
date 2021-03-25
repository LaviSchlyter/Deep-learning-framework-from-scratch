import torch
from torch import nn
from torch.nn import functional as F

# Convolution architecture

# Of course this does not work
class Net1(nn.Module):
    def __init__(self, hidden_layers = 100):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 32, kernel_size= 5)
        #self.conv1 = nn.Conv2d(, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size= 5)
        self.fc1 = nn.Linear(42,hidden_layers)
        self.fc1 = nn.Linear(hidden_layers ,10)


    def forward(self, input_, kernel_size = 5, stride = 2):
        input_ = F.relu(F.max_pool2d(self.conv1(input_), kernel_size = kernel_size, stride =stride))
        input_ = F.relu(F.max_pool2d(self.conv1(input_), kernel_size=kernel_size, stride=stride))
        input_ = F.relu(self.fc1(input_.view(-1, 256)))
        input_ = self.fc2(input_)
        return input_


