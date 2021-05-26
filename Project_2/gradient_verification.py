from architectures import build_shared_model, build_simple_model
from hyper.loss import *
from util import *


def test_gradients(model, loss_func, data, eps=1e-6):
    """
    Print the gradient computed by backpropagation and the gradient computed by numerical differentiation
    side-by-side, for the purposes of debugging GradFn implementation issues.
    """

    # compute the gradients using backpropagation
    for param in model.param():
        param.zero_grad()
    y_pred = model(data.train_x)
    loss = loss_func(y_pred, data.train_y)
    loss.backward()

    for w in model.param():
        w_flat = w.value.flatten()

        for i in range(w_flat.shape[0]):
            w_prev = w_flat[i].item()

            # compute the loss for w[i] + eps
            w_flat[i] = w_prev + eps
            y_pred_plus = model(data.train_x)
            loss_plus = loss_func(y_pred_plus, data.train_y)

            # compute the loss for w[i] - eps
            w_flat[i] = w_prev - eps
            y_pred_minus = model(data.train_x)
            loss_minus = loss_func(y_pred_minus, data.train_y)

            # calculate the numerical gradient as (L(w[i] + eps) - L(w[i] - eps)) / (2 * eps)
            grad_est = (loss_plus.value - loss_minus.value) / (2 * eps)

            # reset the weight to the original value
            w_flat[i] = w_prev

            print(grad_est, "  ", w.grad.flatten()[i])


def main(model, loss_func):
    torch.set_grad_enabled(False)

    data = Data.generate(1000)

    model = build_shared_model()
    loss_func = LossMSE()

    test_gradients(model, loss_func, data)


if __name__ == '__main__':
    main(build_simple_model(), LossMSE())
