""" Test file which must be run from the VM.

Please comment out any model you wish to try
Features include:
- Choosing data size
- Running 6 different models
- Running for multiple rounds
- Logging the loss for each epoch by setting [LOG_EPOCHS = True]
- Plotting the training process and the heatmap of the model's output (both in report)
"""
import torch

from architectures import build_shared_model, build_simple_model
from hyper.loss import LossMSE, LossBCE
from hyper.optimizer import SGD, Adam
from util import run_experiment, set_plot_font_size

ROUNDS = 3
EPOCHS = 250
DATA_SIZE = 1000
LOG_EPOCHS = True
EXTRA_PLOTS = False
MAKE_MOVIE = False


def main_WS_MSE_SGD():
    """ Weight sharing with MSE loss and SGD optimizer """
    run_experiment(
        "WS_MSE_SGD",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=build_shared_model,
        build_optimizer=lambda param: SGD(param, lr=0.4, lambda_=0),
        loss_func=LossMSE(),
        epochs=EPOCHS,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
        make_movie=MAKE_MOVIE,
    )


def main_MSE_SGD():
    """ Using Basic Network with MSE loss and SGD optimizer """
    run_experiment(
        "MSE_SGD",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=build_simple_model,
        build_optimizer=lambda param: SGD(param, lr=0.5, lambda_=1e-4),
        loss_func=LossMSE(),
        epochs=EPOCHS,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
        make_movie=MAKE_MOVIE,
    )


def main_BCE_SGD():
    """ Using Basic Network with BCE loss and SGD optimizer """
    run_experiment(
        "BCE_SGD",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=build_simple_model,
        build_optimizer=lambda param: SGD(param, lr=0.3, lambda_=0),
        loss_func=LossBCE(),
        epochs=EPOCHS,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
        make_movie=MAKE_MOVIE,
    )


def main_MSE_Adam():
    """ Using Basic Network with MSE loss and Adam optimizer """
    run_experiment(
        "MSE_Adam",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=build_simple_model,
        build_optimizer=lambda param: Adam(param),
        loss_func=LossMSE(),
        epochs=EPOCHS,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
        make_movie=MAKE_MOVIE,
    )


def main_BCE_Adam():
    """ Using Basic Network with BCE loss and Adam optimizer """
    run_experiment(
        "BCE_Adam",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=build_simple_model,
        build_optimizer=lambda param: Adam(param),
        loss_func=LossBCE(),
        epochs=EPOCHS,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
        make_movie=MAKE_MOVIE,

    )


def main_WS_MSE_Adam():
    """ Using Basic Network with MSE loss and Adam optimizer """
    run_experiment(
        "WS_MSE_Adam",
        rounds=ROUNDS,
        n=DATA_SIZE,
        build_model=build_shared_model,
        build_optimizer=lambda param: Adam(param),
        loss_func=LossMSE(),
        epochs=EPOCHS,
        log_epochs=LOG_EPOCHS,
        extra_plots=EXTRA_PLOTS,
        make_movie=MAKE_MOVIE,
    )


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    set_plot_font_size()

    main_MSE_SGD()
    # main_BCE_SGD()
    # main_MSE_Adam()
    # main_BCE_Adam()
    # main_WS_MSE_SGD()
    # main_WS_MSE_Adam()
