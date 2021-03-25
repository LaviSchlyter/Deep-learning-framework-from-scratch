
import torch
from torch import nn
from torch.nn import functional as F

# example
#optimizer = torch.optim.SGD(model.parameters(), lr=eta)
def train_model(model, train_input, train_target, mini_batch_size, criterion, optimizer, nb_epochs = 100):

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
