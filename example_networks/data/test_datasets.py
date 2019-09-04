import numpy as np
import math


def load_planar_dataset():

    np.random.seed(1)
    m = 400
    N = int(m / 2)
    D = 2
    X = np.zeros((m, D))
    Y = np.zeros((m, 1), dtype='uint8')
    a = 4

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    X = X.T
    Y = Y.T

    return X / 5, Y


def load_easy_dataset():

    X = [[], []]
    Y = []

    for _ in range(200):

        X[0].append(-.75*np.random.random())
        X[1].append(-.75*np.random.random())
        X[0].append(.75*np.random.random())
        X[1].append(.75*np.random.random())
        Y.append(0)
        Y.append(1)

    return np.array(X), np.array([Y])


def load_easy_dataset_extended():

    X = [[], []]
    Y = []

    for _ in range(100):

        X[0].append(-.75*np.random.random())
        X[1].append(-.75*np.random.random())
        X[0].append(-.75*np.random.random())
        X[1].append(.75*np.random.random())
        X[0].append(.75*np.random.random())
        X[1].append(.75*np.random.random())
        X[0].append(.75*np.random.random())
        X[1].append(-.75*np.random.random())
        Y.append(0)
        Y.append(1)
        Y.append(0)
        Y.append(1)

    return np.array(X), np.array([Y])

