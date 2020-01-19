import numpy as np
import sys
import csv


def read_data(inputfile):
    data = []
    labels = []
    res = []
    with open(inputfile, 'r') as infile:
        csv_reader = csv.reader(infile)
        for row in csv_reader:
            data.append(row)

    for row in data:
        labels.append(int(row[0]))
        x = row[1:]
        x = np.array(x, dtype=np.int32)
        x = np.append(1, x)
        res.append(x)
    res = np.array(res)
    labels = np.array(labels)
    labels.resize(len(labels), 1)
    Y = np.zeros((len(labels), 10))
    for i in range(len(labels)):
        Y[i][int(labels[i])] = 1

    return res.T, labels, Y


def sigmoid(x):
    return 1 / (1 + np.exp(- x))


def softmax(b):
    m = (np.exp(b) / np.sum(np.exp(b)))
    return m


def parameter_initialize(flag, n_x, n_h, n_y):
    if flag == 2:
        A = np.zeros((n_h, n_x + 1))
        B = np.zeros((n_y, n_h + 1))

    if flag != 2:
        A = np.random.uniform(-0.1, 0.1, (n_h, n_x + 1))
        A[:, 0] = 0.0
        B = np.random.uniform(-0.1, 0.1, (n_y, n_h + 1))
        B[:, 0] = 0.0

    B_star = np.delete(B, 0, 1)

    parameter = {"A": A,
                 "B": B,
                 "B_star": B_star}
    return parameter


def forward_propgation(X, parameter):
    A = parameter['A']
    B = parameter['B']
    B_star = parameter['B_star']

    a = np.dot(A, X)
    Z = sigmoid(a)
    Z = np.append(1, Z)
    Z.resize(len(Z), 1)
    b = np.dot(B, Z)
    y_hat = softmax(b)

    cache = {"a": a,
             "Z": Z,
             "b": b,
             "y_hat": y_hat}

    return cache


def loss_function(y_hat, y):
    J = -np.dot(y.T, np.log(y_hat))
    J = np.squeeze(J)
    return float(J)


def back_propagation(parameter, cache, X, Y):
    A = parameter['A']
    B = parameter['B']
    B_star = parameter['B_star']
    a = cache['a']
    Z = cache['Z']
    b = cache['b']
    y_hat = cache['y_hat']

    db = y_hat - Y
    dB = np.dot(db, Z.T)
    Z = np.delete(Z, 0)
    Z.resize(len(Z), 1)
    dZ = np.dot(B_star.T, db)
    da = dZ * Z * (1 - Z)
    dA = np.dot(da, X.T)

    grads = {"dA": dA,
             "dB": dB, }

    return grads


def update_parameters(parameter, grads, learning_rate):
    A = parameter['A']
    B = parameter['B']

    dA = grads['dA']
    dB = grads['dB']

    A -= learning_rate * dA
    B -= learning_rate * dB
    B_star = np.delete(B, 0, 1)

    parameter = {"A": A,
                 "B": B,
                 "B_star": B_star}
    return parameter


def predict(parameter, test_data, test_labels):
    y_predict = np.array([[]])
    for i in range(len(test_labels)):
        X = test_data.T[i]
        cache_test = forward_propgation(X, parameter)
        y_hat = cache_test["y_hat"]
        y_predict = np.append(y_predict, int(np.where(y_hat == np.max(y_hat))[0]))
    y_predict.resize(len(test_labels), 1)
    return y_predict


def error_calculation(y_predict, labels):
    error_rate = 0.0
    for i in range(len(labels)):
        if int(y_predict[i]) != int(labels[i]):
            error_rate += 1 / len(labels)

    print(error_rate)
    return error_rate


if __name__ == '__main__':
    train_in = sys.argv[1]
    test_in = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    rate = float(sys.argv[9])

    train_out_labels = []
    test_out_labels = []
    train_error_rate = 0.0

    train_data, train_labels, y_train = read_data(train_in)
    test_data, test_labels, y_test = read_data(test_in)

    parameter = parameter_initialize(init_flag, 128, hidden_units, 10)

    with open(metrics_out, 'w') as metrics:
        for e in range(num_epoch):
            J_train = 0.0
            J_test = 0.0
            y_predict_train = np.array([[]])
            for i in range(len(train_labels)):
                X = train_data.T[i]
                X = X.reshape(129, 1)
                Y = y_train[i]
                Y = Y.reshape(10, 1)

                cache = forward_propgation(X, parameter)

                grads = back_propagation(parameter, cache, X, Y)

                parameter = update_parameters(parameter, grads, rate)

                y_predict_test = predict(parameter, test_data, test_labels)

                y_hat = cache["y_hat"]
                # print(y_hat.shape)
                J_train += loss_function(y_hat, y_train[i]) / len(train_labels)

                y_predict_train = np.append(y_predict_train, np.where(y_hat == np.max(y_hat))[0])
            print(J_train)

        y_predict_train.resize(len(train_labels), 1)
        y_predict_test = predict(parameter, test_data, test_labels)

        train_error_rate = error_calculation(y_predict_train, train_labels)
        test_error_rate = error_calculation(y_predict_test, test_labels)

        metrics.write("error(train): " + str(train_error_rate) + "\n")
        metrics.write("error(test): " + str(test_error_rate))


def predict(parameter, test_data, test_labels):
    y_predict = np.array([[]])
    for i in range(len(test_labels)):
        X = test_data.T[i]
        cache_test = forward_propgation(X, parameter)
        y_hat = cache_test["y_hat"]
        y_predict = np.append(y_predict, int(np.where(y_hat == np.max(y_hat))[0]))
    y_predict.resize(len(test_labels), 1)
    return y_predict



