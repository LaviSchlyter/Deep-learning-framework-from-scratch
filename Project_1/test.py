import torch
import dlc_practical_prologue as prl
import matplotlib.pyplot as plt
"""
The goal of this project is to implement a deep network such that, given as 
input a series of 2×14×14tensor, corresponding to pairs of 14×14 grayscale images,
it predicts for each pair if the first digit islesser or equal to the second.
"""
Number_of_pairs = 1000
train_input, train_target, train_classes, test_input, test_target, test_classes = prl.generate_pair_sets(Number_of_pairs)


# Write function to test different learning rates
learning_rate, nb_epochs, batch_size = 1e-1, 10, 100

# Make a standardize function
def standardize(input_):
    mean_input = input_.mean()
    std_input = input_.std()
    return [input_.sub_(mean_input).div_(std_input), mean_input, std_input]




"""
Me playing around
image_index = 70 # You may select anything up to 60,000
print(train_input[image_index].size())

print(train_input[image_index][0].size())
output,_ = torch.max(train_input[image_index], 0, keepdim=False)

print(output.size()) # The label is 8

plt.imshow(train_input[image_index][1], cmap='Greys')
plt.show()

# Function will take epochs
"""
