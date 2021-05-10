# Implementation of Autograd
import seaborn as sns
from matplotlib import pyplot as plt

# TODO remove plt and other libraries before handing
# TODO Add a gradient tester
# TODO Organize the files
# TODO Add an optimizer
# TODO plot the ones that are wrongly classified
# TODO Plot a heatmap of the points
from Project_2.modules import *
from Project_2.utils import *


def plot_performance(plot_data, plot_legend, print_loss=True):
    if not print_loss:
        plot_legend = plot_legend[:2]

    for i, legend in enumerate(plot_legend):
        plt.plot(plot_data[:, i], label=legend)

    plt.legend()

    plt.show()


def main():
    # Disable the use of autograd from PyTorch
    torch.set_grad_enabled(False)

    model = Sequential([Linear(2, 50), Relu(0), Linear(50, 25), Relu(0), Linear(25, 1), Sigmoid()])
    # model_ = Sequential([Linear(2 * 10, 50), Relu(), Linear(50, 25), Relu(), Linear(25, 1), Sigmoid()])

    # Generate the set of size n
    n = 1000
    train_input, train_target = generate_disc_set(n)
    test_input, test_target = generate_disc_set(n)


    # Normalizing the data set
    mean, std = train_input.value.mean(), train_input.value.std()
    train_input.value.sub_(mean).div_(std)
    test_input.value.sub_(mean).div_(std)

    train_input_x = train_input[0]
    train_input_y = train_input[1]
    test_input_x = test_input[0]
    test_input_y = test_input[1]

    train_mask = train_target.value[:, 0] > 0.5
    plt.scatter(train_input.value[train_mask, 0], train_input.value[train_mask, 1])
    plt.scatter(train_input.value[~train_mask, 0], train_input.value[~train_mask, 1])
    xmin, xmax, ymin, ymax = plt.axis()

    plt.show()

    loss = LossBCE()
    criterion = Adam(model.param(), alpha=0.0005)
    # criterion = Adam(model.param() + model_.param())
    # criterion = SGD(model.param(), 0.05/n)

    epoch = 400
    plot_data = torch.empty(epoch, 4)
    plot_legend = ["train_error", "test_error", "train_loss", "test_loss"]
    for e in range(epoch):
        criterion.zero_grad()

        # Train evaluation
        y_train = model(train_input)
        # y_train_x = model(train_input_x)
        # y_train_y = model(train_input_y)

        # y_train = model_(y_train_x.cat(y_train_y))

        cost_train = loss(y_train, train_target)
        error_train = evaluate(y_train, train_target)

        # Test evaluation
        # y_test_x = model(test_input_x)
        # y_test_y = model(test_input_y)

        # y_test = model_(y_test_x.cat(y_test_y))
        y_test = model(test_input)
        cost_test = loss(y_test, test_target)
        error_test = evaluate(y_test, test_target)

        # Save values for data plotting
        plot_data[e, 0] = error_train
        plot_data[e, 1] = error_test
        plot_data[e, 2] = cost_train.value / n
        plot_data[e, 3] = cost_test.value / n

        print(format(
            f"For epoch = {e} with {type(loss).__name__} \n Training loss = {format(cost_train.value[0], '.4f')}    ,   Test loss = {format(cost_test.value[0], '.4f')}"))
        cost_train.backward()
        criterion.step()

    image_input_x, image_input_y = torch.meshgrid(torch.linspace(xmin, xmax, 1000), torch.linspace(ymin, ymax, 1000))
    image_input = torch.cat([image_input_x.reshape(-1, 1), image_input_y.reshape(-1, 1)], dim=1)

    image_output = model(Tensor(image_input))
    import matplotlib
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["orange", "yellow", "blue"])
    plt.imshow(image_output.value.reshape(-1, 1000), cmap=cmap)
    # plt.xlim([xmin,xmax])
    # plt.ylim([ymin, ymax])
    plt.show()

    plot_performance(plot_data, plot_legend, True)
    print("Test error: ", evaluate(y_test, test_target))
    y_test_nu = y_test.value.numpy()
    sns.heatmap(y_test_nu)

    print("Train error: ", evaluate(y_train, train_target))

    wrong_pred_coordinates = test_input.value[((y_test.value > 0.5) != (test_target.value > 0.5))[:, 0], :]
    # ok = ((y_test.value > 0.5) != (test_target.value > 0.5))
    # plt.title("Wrongly predicted")
    # plt.scatter(wrong_pred_coordinates[:, 0], wrong_pred_coordinates[:,1])

    # plt.xlim([xmin,xmax])
    # plt.ylim([ymin, ymax])
    plt.show()


if __name__ == '__main__':
    main()
