import numpy as np
import h5py
from matplotlib import pyplot as plt


def load_dataset(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    :param filepath:
    :return:
    """

    train_dataset = h5py.File(filepath + "train_catvnoncat.h5", "r")
    x_train_orig = np.array(train_dataset["train_set_x"][:])  # train set features
    y_train_orig = np.array(train_dataset["train_set_y"][:])  # train set labels

    test_dataset = h5py.File(filepath + "test_catvnoncat.h5", "r")
    x_test_orig = np.array(test_dataset["test_set_x"][:])  # test set features
    y_test_orig = np.array(test_dataset["test_set_y"][:])  # test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    y_train_orig = y_train_orig.reshape((1, y_train_orig.shape[0]))
    y_test_orig = y_test_orig.reshape((1, y_test_orig.shape[0]))

    return x_train_orig, y_train_orig, x_test_orig, y_test_orig, classes


def sigmoid(x: np.ndarray) -> np.ndarray:
    """

    :param x:
    :return:
    """

    return 1 / (1 + np.exp(-x))


def forward_propagation(parameters: dict, x: np.ndarray, y: np.ndarray) -> float:
    """

    :param parameters:
    :param x:
    :param y:
    :return:
    """

    w, b = parameters["w"], parameters["b"]

    z = np.dot(w.T, x) + b
    # parameters["z"] = z
    a = sigmoid(z)
    parameters["a"] = a
    cost = (-1) / x.shape[1] * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
    return cost


def backward_propagation(parameters: dict, x: np.ndarray, y: np.ndarray):
    """

    :param parameters:
    :param x:
    :param y:
    :return:
    """

    a = parameters["a"]
    dz = a - y
    dw = 1 / x.shape[1] * np.dot(x, dz.T)
    db = 1 / x.shape[1] * np.sum(dz, axis=1, keepdims=True)
    parameters["dz"] = dz
    parameters["dw"] = dw
    parameters["db"] = db
    return parameters


def gradient_descent(parameters: dict, alpha: float, number_of_iterations: int, x: np.ndarray, y: np.ndarray, print_cost: bool = False):
    """

    :param parameters:
    :param alpha:
    :param number_of_iterations:
    :param x:
    :param y:
    :return:
    """

    costs = []

    for i in range(number_of_iterations):
        cost = forward_propagation(parameters, x, y)
        backward_propagation(parameters, x, y)

        w = parameters["w"]
        b = parameters["b"]
        dw = parameters["dw"]
        db = parameters["db"]

        w = w - alpha * dw
        b = b - alpha * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

        parameters["w"] = w
        parameters["b"] = b

    return parameters, costs


def predict(w: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """

    :param w:
    :param b:
    :param x:
    :return:
    """

    z = np.dot(w.T, x) + b
    y_predict = sigmoid(z)

    return y_predict


if __name__ == "__main__":
    file_path = "dataset/"
    x_train, y_train, x_test, y_test, class_list = load_dataset(file_path)

    x_train_flatten = x_train.reshape(x_train.shape[0], -1).T
    x_test_flatten = x_test.reshape(x_test.shape[0], -1).T

    # shape tests
    assert (x_train_flatten.shape == (12288, 209))
    assert (y_train.shape == (1, 209))
    assert (x_test_flatten.shape == (12288, 50))
    assert (y_test.shape == (1, 50))
    # end of test

    x_train_norm = x_train_flatten / 255.
    x_test_norm = x_test_flatten / 255.

    # sigmoid test
    assert (sigmoid(np.array([0, 2])).all() == np.array([0.5, 0.88079708]).all())
    # end of test

    # forward and backward propagation test
    w_test, b_test, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
    params = {"w": w_test, "b": b_test}
    cost = forward_propagation(parameters=params, x=X, y=Y)
    params = backward_propagation(parameters=params, x=X, y=Y)
    assert (params["dw"].all() == np.array([[0.99845601], [2.39507239]]).all())
    assert (params["db"] == 0.001455578136784208)
    assert (cost == 5.801545319394553)
    # end of test

    size_of_hidden_unit = 1
    w = np.zeros((x_train_norm.shape[0], size_of_hidden_unit))
    b = 0

    parameters = {"w": w, "b": b}
    cost = forward_propagation(parameters, x_train_norm, y_train)
    parameters = backward_propagation(parameters, x_train_norm, y_train)
    assert (parameters["dw"].shape == w.shape)
    assert (parameters["db"].dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    # gradientdescent test
    w_test, b_test, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
    params = {"w": w_test, "b": b_test}
    params, cost_test = gradient_descent(params, x=X, y=Y, number_of_iterations=100, alpha=0.009, print_cost=False)

    assert (params["w"].all() == np.array([[0.19033591], [0.12259159]]).all())
    assert (params["b"] == 1.9253598300845747)
    assert (params["dw"].all() == np.array([[0.67752042], [1.41625495]]).all())
    assert (params["db"] == 0.21919450454067652)
    # end of test

    alpha = 0.005
    parameters, costs = gradient_descent(parameters, alpha, 2000, x_train_norm, y_train, True)
    plt.plot(costs)
    plt.show()

    # predict test
    w_test = np.array([[0.1124579], [0.23106775]])
    b_test = np.array([-0.3])
    X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
    predict_test = predict(w_test, b_test, X)
    predict_test = (predict_test >= 0.5) * 1.
    assert (predict_test.all() == np.array([[1., 1., 0.]]).all())
    # end of test

    y_predict = predict(parameters["w"], parameters["b"], x_test_norm)

    y_train_predict = predict(parameters["w"], parameters["b"], x_train_norm)
    y_train_predict = (y_train_predict >= 0.5) * 1.0

    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_train_predict - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_predict - y_test)) * 100))
