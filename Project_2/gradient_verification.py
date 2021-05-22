from modules import *
from loss import *
from util import *
from optimizer import *


def basic_network_1():
    return Sequential([
        Linear(2, 1),
    ])


def basic_network_WS_1():
    return Sequential([
        Linear(1, 10),
        Tanh(),
        Linear(10, 25),
        Tanh(),


    ])


def basic_network_WS_2():
    return Sequential([
        Linear(50, 40),
        Relu(),
        Linear(40, 25),
        Relu(),
        Linear(25, 1),
        Sigmoid()
    ])


def test_gradients(model, loss_func, data, eps=1e-6):
    Adam(model.param()).zero_grad()
    y_pred = model(data.train_x)
    loss = loss_func(y_pred, data.train_y)
    loss.backward()
    MSE = 0
    for w in model.param():
        for i in range(w.value.flatten().shape[0]):
            w_prev = w.value.flatten()[i].item()
            w.value.flatten()[i] = w_prev + eps
            y_pred_plus = model(data.train_x)
            loss_plus = loss_func(y_pred_plus, data.train_y)
            w.value.flatten()[i] = w_prev - eps
            y_pred_minus = model(data.train_x)
            loss_minus = loss_func(y_pred_minus, data.train_y)

            grad_est = (loss_plus.value - loss_minus.value) / (2 * eps)
            w.value.flatten()[i] = w_prev
            print(grad_est, "  ", w.grad.flatten()[i])



def main():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)

    model = basic_network_1()

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)

    # Normalizing the data set
    train_input, test_input = normalize(train_input, test_input)

    # Store in Data structure
    data = Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)

    #loss_func = LossBCE()
    loss_func = LossMSE()

    optimizer = Adam(model.param())
    # optimizer = SGD(model.param(), 9/n)
    # plot_data, plot_legend = train_model(model, optimizer, loss_func, data, epoch=400, log_loss=False)

    test_gradients(model, loss_func, data)


def main_share():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)

    model_1 = basic_network_WS_1()
    model_2 = basic_network_WS_2()

    model = WeightSharing(model_1, model_2)

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)

    # Normalizing the data set
    train_input, test_input = normalize(train_input, test_input)

    # Store in Data structure
    data = Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)

    loss_func = LossBCE()
    # loss_func = LossMSE()

    optimizer = Adam(model.param())
    # optimizer = SGD(model.param(), 0.03 / n)

    plot_data, plot_legend = train_model(model, optimizer, loss_func, data, epoch=400, log_epochs=False)
    test_gradients(model, loss_func, data)
    # test_gradients(model, loss_func, data)

    y_test = model(data.test_x)
    y_train = model(data.train_x)


if __name__ == '__main__':
    main()
    #print("---------------------------------Sharing----------------------------------- ")
    #main_share()
