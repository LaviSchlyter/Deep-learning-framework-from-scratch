import torch
from torch import Tensor
import dlc_practical_prologue as prl


n = 10
d = 2
train_x = torch.rand(n,d )
train_target = torch.rand(n)
x = torch.rand(d)

# l2 norm is the sqaure root of the sum of the squared difference


train_data, train_target, test_data, test_target = prl.load_data(cifar = None, one_hot_labels = False, normalize = False, flatten = True)


def nearest_classification(train_x, train_y, x):
    min_index = torch.pow(train_x - x, 2).sum(1).sort()[1][0].item()
    return train_y[min_index].item()


nearest_classification(train_data, train_target, test_data)


def compute_nb_errors(train_input, train_target, test_input, test_target, mean=None, proj=None):
    if mean is not None:
        train_input -= mean
        test_input -= mean

    if proj is not None:
        train_input = train_input.mm(proj)
        test_input = test_input.mm(proj)

    num_errors = 0
    for i in range(train_data[0].size()[0]):
        x = test_data[i][:]  # Equivalent to test_data.narrow(0,i,1)
        if nearest_classification(train_input, train_target, x) != test_target[i]:
            num_errors += 1

    return num_errors


compute_nb_errors(train_data, train_target, test_data, test_target, mean=None, proj=None)


def PCA(x):  # x = nxd
    mean = x.mean(dim=0)
    x_ = x - mean

    v, vec = torch.eig(x_.t().mm(x_))
    t = v.narrow(1, 0, 1).reshape((v.shape[0]))
    _, indices = torch.sort(t, descending=True)

    # v = v.narrow(1,0,1)
    # dim = v.size()[0]
    # eig_sort = v.sort()[0][:dim]

    return [mean, vec[indices]]

cifar_train, cifar_train_target, cifar_test, cifar_test_target = prl.load_data(cifar=True)


def generate_data(n, dim, num_labels):  # Number of labels

    x_train = torch.randn(n, dim)
    y_train = torch.randn(num_labels)

    x_test = torch.randn(n, dim)
    y_test = torch.randn(num_labels)

    return x_train, y_train, x_test, y_test


#def reduction_dim(basis, reduct_dim):
    #new_basis =

def test_nearest(train, target_train, test, target_test, dim_red=None):
    if dim_red is not None:  # Look at PCA
        train_mean, train_basis = PCA(train)





