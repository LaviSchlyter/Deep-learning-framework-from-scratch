# Implementation of Autograd

from loss import *
from modules import *
from optimizer import *
from util import *

ROUNDS = 3
DATA_SIZE = 1000
LOG_EPOCHS = False
EXTRA_PLOTS = True


# TODO Video
# TODO make simple architectures
# TODO Output plot reports (labels, legends, no title)
# TODO maybe plot input gradient instead of just the evaluation in the heatmap plot

def network_WS_1():
    return Sequential([
        Linear(1, 10),
        Relu(),
        Linear(10, 25),
        Relu(),
        Linear(25, 10),
    ])


def network_WS_2():
    return Sequential([
        Linear(20, 40),
        Relu(),
        Linear(40, 25),
        Relu(),
        Linear(25, 1),
        Sigmoid()
    ])


def network_1():
    return Sequential([
        Linear(2, 25),
        Relu(),
        Linear(25, 25),
        Relu(),
        Linear(25, 25),
        Relu(),
        Linear(25, 1),
        Sigmoid()

    ])


def network_2():
    return Sequential([
        Linear(2, 25),
        Tanh(),
        Linear(25, 25),
        Tanh(),
        Linear(25, 1),
        Tanh(),
        Linear(25, 1),
        Sigmoid()

    ])


def network_3():
    return Sequential([
        Linear(2, 25),
        Sigmoid(),
        Linear(25, 25),
        Sigmoid(),
        Linear(25, 1),
        Sigmoid(),
        Linear(25, 1),
        Sigmoid()

    ])


def build_shared_model():
    model_1 = network_WS_1()
    model_2 = network_WS_2()

    # Combining the models
    model = WeightSharing(model_1, model_2)
    return model


def main_WS_MSE_SGD():
    run_experiment(
        "WS_MSE_SGD",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=build_shared_model,
        build_optimizer=lambda param: SGD(param, lr=0.3 / DATA_SIZE, lambda_=0),
        loss_func=LossMSE(),
        epochs=250,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
    )


def main_MSE_SGD():
    run_experiment(
        "MSE_SGD",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=network_1,
        build_optimizer=lambda param: SGD(param, 0.3 / DATA_SIZE, lambda_=0.1),
        loss_func=LossMSE(),
        epochs=250,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
    )


def main_BCE_SGD():
    run_experiment(
        "BCE_SGD",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=network_1,
        build_optimizer=lambda param: SGD(param, 0.3 / DATA_SIZE, lambda_=0),
        loss_func=LossBCE(),
        epochs=250,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
    )


def main_MSE_Adam():
    run_experiment(
        "MSE_Adam",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=network_1,
        build_optimizer=lambda param: Adam(param),
        loss_func=LossMSE(),
        epochs=250,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
    )


def main_BCE_Adam():
    run_experiment(
        "BCE_Adam",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=network_1,
        build_optimizer=lambda param: Adam(param),
        loss_func=LossBCE(),
        epochs=250,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
    )


def main_WS_MSE_Adam():
    run_experiment(
        "WS_MSE_Adam",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=build_shared_model,
        build_optimizer=lambda param: Adam(param),
        loss_func=LossMSE(),
        epochs=250,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
    )


if __name__ == '__main__':
    torch.set_grad_enabled(False)

    main_MSE_SGD()
    main_BCE_SGD()
    main_MSE_Adam()
    main_BCE_Adam()
    main_WS_MSE_SGD()
    main_WS_MSE_Adam()
