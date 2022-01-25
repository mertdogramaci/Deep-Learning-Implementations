import numpy as np


def sigmoid(z: np.ndarray):
    """
    sigmoid(z) = 1 / (1 + e^-z)

    :param z:
    :return:
    """

    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(x: np.ndarray):
    """
    sigmoid_derivative(x) = sigmoid(x) * (1 - sigmoid(x))

    :param x:
    :return:
    """

    return sigmoid(x) * (1 - sigmoid(x))


def image2vector(image: np.ndarray):
    vector = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return vector


def normalize_rows(x: np.ndarray):
    """
    x_normalized = x / (Σ (x_j)^2)^(1/2)

    :param x:
    :return:
    """

    x_square = np.square(x)
    x_sum = np.sum(x_square, axis=1, keepdims=True)
    x_sqrt = np.sqrt(x_sum)

    x_norm = x.copy()
    x_norm = np.divide(x_norm, x_sqrt)

    return x_norm


def softmax(x: np.ndarray):
    """
    softmax(x) = e^x / Σ e^x_j

    :param x:
    :return:
    """

    x_exp = np.exp(x)
    x_sum = x_exp.sum(axis=1, keepdims=True)
    return x_exp / x_sum


def L1(y_hat: np.ndarray, y: np.ndarray):
    """
    Mean Absolute Error Loss
    L1(y, y') = Σ |y_i - y'_i|

    :param y_hat:
    :param y:
    :return:
    """

    return sum(abs(y_hat - y))


def L2(y_hat: np.ndarray, y: np.ndarray):
    """
    Mean Square Error Loss
    L2(y, y') = Σ (y_i - y'_i)^2

    :param y_hat:
    :param y:
    :return:
    """

    return sum(np.square(y_hat - y))


if __name__ == "__main__":
    # Testing sigmoid and sigmoid derivative functions
    arr = np.array([1, 2, 3])
    print(sigmoid(arr))
    print(sigmoid_derivative(arr))

    # Testing image2vector function
    example_image = np.array([[[0.67826139, 0.29380381],
                               [0.90714982, 0.52835647],
                               [0.4215251, 0.45017551]],

                              [[0.92814219, 0.96677647],
                               [0.85304703, 0.52351845],
                               [0.19981397, 0.27417313]],

                              [[0.60659855, 0.00533165],
                               [0.10820313, 0.49978937],
                               [0.34144279, 0.94630077]]])
    print("image2vector(image) = " + str(image2vector(example_image)))

    # Testing normalizeRows function
    arr_2 = np.array([[0, 3, 4], [1, 6, 4]])
    print("normalizeRows(x) = " + str(normalize_rows(arr_2)))

    # Testing softmax function
    arr3 = np.array([[9, 2, 5, 0, 0],
                     [7, 5, 0, 0, 0]])
    print("softmax(x) = " + str(softmax(arr3)))

    # Testing Mean Absolute Error Loss
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L1 = " + str(L1(yhat, y)))

    # Testing Mean Square Error Loss
    yhat = np.array([.9, 0.2, 0.1, .4, .9])
    y = np.array([1, 0, 0, 1, 1])
    print("L2 = " + str(L2(yhat, y)))
