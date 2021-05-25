import math
import os
import matplotlib.pyplot as plt
import torch
from core import HyperCube

# TODO Figure out what to do with the plotting part

def set_plot_font_size():
    """Set larger font sizes for pyplot."""
    params = { "legend.fontsize": "large", "axes.labelsize": "large", "axes.titlesize": "large", "xtick.labelsize": "large",
               "ytick.labelsize": "large" }
    plt.rcParams.update(params)

def generate_disc_set(nb):
    """ Generate a disk of radius 1/sqrt(2*pi)

    :param nb: Number of points wanted
    :return: HyperCube with the x coordinates and with the y targets
    """
    input_ = torch.empty(nb, 2).uniform_()
    target = ((input_ - 0.5).pow(2).sum(1) < 1 / (2 * math.pi)).float()
    return HyperCube(input_), HyperCube(target[:, None])


def evaluate_error(pred, target):
    """ Compute the error rate

    :param pred: HyperCube with the predicted values from model
    :param target: HyperCube with the target values
    :return: The error rate between target and predicted
    """
    return 1 - ((pred.value > 0.5) == (target.value > 0.5)).sum() / len(pred.value)


def normalize(train_x, test_x):
    """ Normalize the data

    We retrieve the mean and std of the training data and apply normalization on both the training and test data
    :param train_x: HyperCube of training data
    :param test_x:HyperCube of test data
    :return: HyperCubes normalized
    """
    mean, std = train_x.value.mean(), train_x.value.std()
    train_x.value.sub_(mean).div_(std)
    test_x.value.sub_(mean).div_(std)
    return train_x, test_x


class Data:
    def __init__(self, train_x, train_y, test_x, test_y):
        self.test_y = test_y
        self.test_x = test_x
        self.train_y = train_y
        self.train_x = train_x

    @classmethod
    def generate(cls, n: int):
        """

        :param n: Number of data to generate
        :return: a Data struct containing the train, test inputs and targets
        """
        train_input, train_target = generate_disc_set(n)
        test_input, test_target = generate_disc_set(n)
        train_input, test_input = normalize(train_input, test_input)
        return Data(train_x=train_input, train_y=train_target, test_x=test_input, test_y=test_target)


PLOT_LEGEND = ["Train_error", "Test_error", "Train_loss", "Test_loss"]


def train_model(model, optimizer, loss_func, data, epoch, log_epochs):

    plot_data = torch.empty(epoch, len(PLOT_LEGEND))

    for e in range(epoch):

        optimizer.zero_grad()

        y_train = model(data.train_x)
        cost_train, error_train = evaluate_model(y_train, data.train_y, loss_func)

        y_test = model(data.test_x)
        cost_test, error_test = evaluate_model(y_test, data.test_y, loss_func)

        # Save values for data plotting
        plot_data[e, 0] = error_train
        plot_data[e, 1] = error_test
        plot_data[e, 2] = cost_train.value
        plot_data[e, 3] = cost_test.value

        cost_train.backward()
        optimizer.step()

        if log_epochs:
            # Printing the losses
            loss_name = type(loss_func).__name__
            print(
                f"For Epoch = {e} with {loss_name}: "
                f"Train_loss = {cost_train.item():.4f}, "
                f"Train_error={error_train:.4f}, "
                f"Test_loss = {cost_test.item():.4f}, "
                f"Test_error={error_test:.4f}"
            )

    return plot_data


def evaluate_model(pred_y, true_y, loss_func):
    """ Evaluating both the loss and the accuracy (error)

    :param pred_y: HyperCube of predicted values
    :param true_y: HyperCube of target values
    :param loss_func: Loss function used
    :return: The cost and the error
    """
    cost = loss_func(pred_y, true_y)
    error = evaluate_error(pred_y, true_y)
    return cost, error


def run_experiment(
        name: str, rounds: int, n: int,
        build_model, build_optimizer, loss_func, epochs, log_epochs: bool,
        extra_plots: bool
):
    """

    :param name: Name of the experiment :str
    :param rounds: Number of rounds : int
    :param n: Number of data that is to be generated : int
    :param build_model: Network used
    :param build_optimizer:Optimizer used
    :param loss_func: Loss function used
    :param epochs: Number of epochs:int
    :param log_epochs: Whether to print the losses or not : Bool
    :param extra_plots: Whether to print final plots or not : Bool
    """
    print(f"Running experiment {name}")

    all_plot_data = torch.zeros(rounds, epochs, len(PLOT_LEGEND))

    model, data = None, None

    for round in range(rounds):
        print(f"Round {round + 1}/{rounds}")

        data = Data.generate(n)

        model = build_model()
        optimizer = build_optimizer(model.param())

        plot_data = train_model(model, optimizer, loss_func, data, epochs, log_epochs)
        all_plot_data[round, :, :] = plot_data

    plot_experiment(name, model, data, all_plot_data, extra_plots)


def plot_experiment(name: str, model, data, all_plot_data, extra_plots):
    os.makedirs(f"data/{name}", exist_ok=True)

    plot_data_std, plot_data_mean = torch.std_mean(all_plot_data, dim=0)

    # Print the final test/train error
    print(f"Train error: {plot_data_mean[-1, 0]:.4f} +- {plot_data_std[-1, 0]:.4f}")
    print(f"Test error: {plot_data_mean[-1, 1]:.4f} +- {plot_data_std[-1, 1]:.4f}")

    # Training plot
    fig, ax = plt.subplots(1)
    ax.plot(plot_data_mean)

    if len(all_plot_data) > 1:
        ax.set_prop_cycle(None)
        ax.plot(plot_data_mean + plot_data_std, '--', alpha=.5)
        ax.set_prop_cycle(None)
        ax.plot(plot_data_mean - plot_data_std, '--', alpha=.5)

    ax.legend(PLOT_LEGEND)
    ax.set_xlabel("Epoch")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_ylabel("Error")


    fig.savefig(f"data/{name}/training.png")
    fig.show()

    if extra_plots:
        # scatter plot
        fig = plt.figure(figsize=(7, 7))

        y_train_pred = model(data.train_x)
        y_test_pred = model(data.test_x)
        y_test_bool_pred = (y_test_pred.value > 0.5)
        boolean_falsely_predicted = (y_test_bool_pred != data.test_y.value)
        coordinate_false = data.test_x.value[boolean_falsely_predicted.squeeze(1)]
        print("coord", coordinate_false.shape)

        y_train_bool_pred = (y_train_pred.value > 0.5).squeeze(1)
        plt.scatter(data.train_x.value[y_train_bool_pred, 0], data.train_x.value[y_train_bool_pred, 1], color="red")
        plt.scatter(data.train_x.value[~y_train_bool_pred, 0], data.train_x.value[~y_train_bool_pred, 1], color="blue")


        xmin, xmax, ymin, ymax = plt.axis()
        fig.savefig(f"data/{name}/distribution_points.png")
        #fig.show()

        # heatmap
        image_input_x, image_input_y = torch.meshgrid(
            torch.linspace(xmin, xmax, 1000),
            torch.linspace(ymin, ymax, 1000)
        )
        image_input = torch.cat([image_input_x.reshape(-1, 1), image_input_y.reshape(-1, 1)], dim=1)
        image_output = model(HyperCube(image_input))
        import matplotlib
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", [(0,"blue"), (0.5,"yellow"), (1,"red")])
        fig = plt.figure(figsize=(7, 7))


        plt.imshow(image_output.value.reshape(-1, 1000).rot90(k=1), cmap=cmap, extent= (xmin, xmax,ymin, ymax))
        circle1 = plt.Circle((0, 0), math.sqrt(6)/math.sqrt(math.pi), color='black', fill=False)
        plt.gca().add_patch(circle1)
        plt.scatter(coordinate_false[:, 0], coordinate_false[:, 1], 40, color="black", marker="x")
        fig.savefig(f"data/{name}/density_correct.png")
        plt.show()
