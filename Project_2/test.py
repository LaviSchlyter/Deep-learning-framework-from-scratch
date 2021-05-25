""" Test file which must be run from the VM.

Please comment out any model you wish to try
Features include:
- Choosing data size
- Running 6 different models
- Running for multiple rounds
- Logging the loss for each epoch by setting [LOG_EPOCHS = True]
- Plotting the training process and the heatmap of the model's output (both in report)
"""

from architectures import *
ROUNDS = 1
DATA_SIZE = 1000
LOG_EPOCHS = True
EXTRA_PLOTS = True

if __name__ == '__main__':
    set_plot_font_size()
    torch.set_grad_enabled(False)
    main_MSE_SGD()
    #main_BCE_SGD()
    #main_MSE_Adam()
    #main_BCE_Adam()
    #main_WS_MSE_SGD()
    #main_WS_MSE_Adam()
