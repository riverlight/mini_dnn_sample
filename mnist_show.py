# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def main():
    mnist_file = "./csv/mnist_train_100.csv"
    with open(mnist_file, "rt") as fp:
        lines = fp.readlines()
    print(lines[0][0])
    lst_v = lines[0].split(',')
    image_array = np.asfarray(lst_v[1:]).reshape((28, 28))
    plt.imshow(image_array, cmap='Greys', interpolation='None')
    plt.show()

if __name__ == "__main__":
    main()
