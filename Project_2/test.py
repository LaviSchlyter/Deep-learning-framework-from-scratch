import torch
import math
import matplotlib.pyplot as plt
import seaborn as sns


n = 1000
# Generate the set
def generate_disc_set(nb):
    input = torch.empty(nb, 2).uniform_(0, 1)
    target = (input- 0.5).pow(2).sum(1).sub(1 / (2*math.pi)).sign().add(1).div(2).long()
    return input, target

train_input, train_target = generate_disc_set(n)
test_input, test_target = generate_disc_set(n)
plt.figure(figsize = (7,7))
scat = sns.scatterplot(x = train_input[:,0], y=train_input[:,1], hue = train_target)
plt.show()