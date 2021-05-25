# File containing the different architectures tested

from loss import *
from modules import *
from util import *
from optimizer import *
from test import ROUNDS, DATA_SIZE, LOG_EPOCHS, EXTRA_PLOTS

def network_WS_1():
    """ First part of the Weight Sharing network
    """
    return Sequential([
        Linear(1, 10),
        Relu(),
        Linear(10, 25),
        Relu(),
    ])


def network_WS_2():
    """ Second part of the Weight Sharing network

    The input layer of the second part of the WS network must be two times the end layer of the first part of the network.
    Else dimensions do not match
    :return: Sequential
    """
    return Sequential([
        Linear(50, 40),
        Relu(),
        Linear(40, 25),
        Relu(),
        Linear(25, 1),
        Sigmoid()
    ])


def basic_network():
    return Sequential([
        Linear(2, 25),
        Tanh(),
        Linear(25, 25),
        Relu(),
        Linear(25, 25),
        Relu(),
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
    """ Weight sharing with MSE loss and SGD optimizer

    """
    run_experiment(
        "WS_MSE_SGD",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=build_shared_model,
        build_optimizer=lambda param: SGD(param, lr=0.3, lambda_=0),
        loss_func=LossMSE(),
        epochs=250,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
    )


def main_MSE_SGD():
    """ Using Basic Network with MSE loss and SGD optimizer
    """
    run_experiment(
        "MSE_SGD",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=basic_network,
        build_optimizer=lambda param: SGD(param, lr=0.3, lambda_=1e-4),
        loss_func=LossMSE(),
        epochs=250,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
    )


def main_BCE_SGD():
    """ Using Basic Network with BCE loss and SGD optimizer
    """
    run_experiment(
        "BCE_SGD",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=basic_network,
        build_optimizer=lambda param: SGD(param, lr=0.3, lambda_=0),
        loss_func=LossBCE(),
        epochs=250,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
    )


def main_MSE_Adam():
    """ Using Basic Network with MSE loss and Adam optimizer
    """
    run_experiment(
        "MSE_Adam",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=basic_network,
        build_optimizer=lambda param: Adam(param),
        loss_func=LossMSE(),
        epochs=250,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
    )


def main_BCE_Adam():
    """ Using Basic Network with BCE loss and Adam optimizer
    """
    run_experiment(
        "BCE_Adam",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=basic_network,
        build_optimizer=lambda param: Adam(param),
        loss_func=LossBCE(),
        epochs=250,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
    )


def main_WS_MSE_Adam():
    """ Using Basic Network with MSE loss and Adam optimizer
        """
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

