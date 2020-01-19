import sys
import numpy as np
import time

def sigmoid(s):
    # print(s)
    # time.sleep(0.1)
    return 1 / (1 + np.exp(-s))

def find_lable(s):
    if sigmoid(s) > 0.5:
        return 1
    else:
        return 0

def sparse_dot(X, W):
    res = 0.0
    for k, v in X.items():
        res += (v * W[int(k)])
    return res

def sparse_gradient(X, delta, length):
    gradient = np.zeros((length, 1))
    for k, v in X.items():
        gradient[int(k)] = v * delta
    return gradient


def train(num_epochs, train_features, train_labels, weight, learning_rate):

    for j in range(0, num_epochs):
        for i in range(0, len(train_features)):
            z = sparse_dot(train_features[i], weight)
            delta_update = (sigmoid(z) - train_labels[i])
            weight_gradient = sparse_gradient(train_features[i], delta_update, len(weight))
            weight = np.subtract(weight, learning_rate * weight_gradient)

    return weight

def predict(features, weight, actual_label):
    label = []
    b = []
    for m in features:
        s = sparse_dot(m, weight)
        sign = find_lable(s)
        label.append(sign)
    for i in range(len(label)):
        if label[i] != actual_label[i]:
            b.append(label[i])
    error_rate = float(len(b)/len(label))
    error_rate = round(error_rate, 6)
    return label, error_rate

def readfile(file_input):
    with open(file_input, "r") as infile:
        file_line = infile.readlines()
    label = []
    dict_list = []
    for line in file_line:
        temp_dict = {}
        a = line.split("\t")
        label.append(int(a[0]))
        for i in range(1, len(a)):
            b = a[i].split(":")
            temp_dict[b[0]] = int(b[1])
        dict_list.append(temp_dict)
    return dict_list, label

def readtrain(file_input):
    with open(file_input, "r") as infile:
        file_line = infile.readlines()
    label = []
    dict_list = []
    for line in file_line:
        temp_dict = {}
        a = line.split("\t")
        label.append(int(a[0]))
        for i in range(1, len(a)):
            b = a[i].split(":")
            temp_dict[b[0]] = int(b[1])
        dict_list.append(temp_dict)
    for dict_ele in dict_list:
        dict_ele[str(len(weight)-1)] = 1
    return dict_list, label

def write_file(label, f_out):
    new_label = ""
    for label_ele in label:
        new_label += str(label_ele) + "\n"
    with open(f_out, 'w') as out_file:
        out_file.write(new_label)

start = time.clock()
if __name__ == "__main__":

    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])

    learning_rate = 0.1

    with open(dict_input, "r")as infile:
        dict_line = infile.readlines()
    weight = np.zeros((len(dict_line) + 1, 1))

    train_dict, train_label = readtrain(train_input)
    validation_dict, validation_label = readfile(validation_input)
    test_dict, test_label = readfile(test_input)

    learned_weight = train(num_epoch, train_dict, train_label, weight, learning_rate)

    new_train_label, trainError = predict(train_dict, learned_weight, train_label)
    new_test_label, testError = predict(test_dict, learned_weight, test_label)

    write_file(new_train_label, train_out)
    write_file(new_test_label, test_out)


    with open(metrics_out, 'w') as outfile:
        outfile.write('error(train): ' + str(trainError) + '\n')
        outfile.write('error(test): ' + str(testError))

    end = time.clock()
    print("\nRuntime:"+str(end-start)+'s')