import os

from matplotlib import pyplot
from mpl_toolkits.axes_grid1 import ImageGrid

from util import load_data


def main():
    rows = 10
    cols = 10

    data = load_data(rows)
    data.expand_train_transform(cols)

    fig = pyplot.figure()
    grid = ImageGrid(fig, 111, (rows, cols))

    for r in range(rows):
        for c in range(cols):
            i = c * rows + r

            ax = grid.axes_row[r][c]
            ax.imshow(data.train_x[i, 0, :, :])

    os.makedirs("output")
    pyplot.savefig("output/expanded_data.png")
    pyplot.show()


if __name__ == '__main__':
    main()
