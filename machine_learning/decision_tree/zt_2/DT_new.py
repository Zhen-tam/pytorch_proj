import sys
import csv
from math import log
import numpy as np
from collections import Counter


class Node:
    def __init__(self, label, val):
        self.left = None
        self.right = None
        self.label = label
        self.val = val
        self.dist = {}

    # Function to count number of descendants
    def getDist(self, data, class_list):
        for i in class_list:
            self.dist[i] = 0
        # print(self.dist)
        for value in data:
            self.dist[value[-1]] += 1
        # print(self.dist)

    # def getDist(self,data,class_list):
    #     labels_list_tmp = []
    #     for item in data:
    #         labels_list_tmp.append(item[-1])
    #         self.dist = dict(Counter(labels_list_tmp))

    # Function for each leaf to make classification decision using majority vote
    def classify(self):
        dist = dict(self.dist)
        # k = list(dist.keys())
        # v = list(dist.values())
        # print(k[v.index(max(v))])
        return max(dist, key=dist.get)

    def printNode(self, level):
        if self.left:
            self.left.printNode(level + 1)
        if self.right:
            self.right.printNode(level + 1)
        print("| " * level, end='\b')
        if self.label:
            print(self.label, end=" = ")
        if self.val:
            print(self.val, end=': ')
        str_dist = str(self.dist).replace('{', '[')
        str_dist = str_dist.replace('}', ']')
        str_dist = str_dist.replace(", ", "/ ")
        str_dist = str_dist.replace('\'', '')
        str_dist = str_dist.replace(':', '')
        print(str_dist)
        # if self.left:
        #     self.left.printNode(level + 1)
        # if self.right:
        #     self.right.printNode(level + 1)


# Function to compute entropy
def calEntropy(data):
    prob = {}
    for value in data:
        if value[-1] not in prob:
            prob[value[-1]] = 1.0
        else:
            prob[value[-1]] += 1.0
    entropy = 0
    for i in prob:
        entropy -= prob[i] / len(data) * log(prob[i] / len(data), 2)
    return entropy


# Split the data based on given attribute(index) and its value
# def splitData(data, attribute, value):
#     new_data = []
#     for instance in data:
#         if instance[attribute] == value:
#             new_instance = instance[:attribute]
#             new_instance.extend(instance[attribute + 1:])
#             new_data.append(new_instance)
#     return new_data # split date according to attribute

def splitData(data, attribute, value):
    data = np.array(data)
    new_data = data[data[:, attribute] == value, :]
    new_data = np.delete(new_data, attribute, axis = 1)
    new_data = list(new_data)
    return new_data


# Function to choose the best attribute to branch
# Compute mutual infomation for each attribute.
# Branch on the one with highest mutual infomation
def chooseBranch(data):
    attribute_num = len(data[0]) - 1
    entropy = calEntropy(data)
    max_MI = 0
    attribute = -1
    for i in range(attribute_num):
        label_list = [instance[i] for instance in data]
        value_set = set(label_list)  # set of possible options for current attribute
        conditional_ent = 0
        for value in value_set:
            # split data for each value and calculate specific conditional entropy
            temp_data = splitData(data, i, value)
            conditional_ent += len(temp_data) / float(len(data)) * calEntropy(temp_data)
        if entropy - conditional_ent > max_MI:
            max_MI = entropy - conditional_ent
            attribute = i
    return attribute


# # Function to count number of descendants
# def Counter(data, class_list):
#     count = {}
#     for i in class_list:
#         count[i] = 0
#     for value in data:
#         count[value[-1]] += 1
#     return str(count)


# Function to create sub-trees of node
def growTree(node, data, labels, class_list, tree_depth, max_depth):
    node.getDist(data, class_list)

    new_class_list = [instance[-1] for instance in data]
    if new_class_list.count(class_list[0]) == len(new_class_list):
        # perfectly classified
        # node.getDist(data, class_list)
        return

    if len(data[0]) == 1:
        # no attribute to be branch. leaf
        # node.getDist(data, class_list)
        return

    if tree_depth >= max_depth:
        return

    # Branch
    # pdb.set_trace()
    branch_ind = chooseBranch(data)
    if branch_ind == -1:
        # mutual infomation is 0, stop
        return
    curr_label = labels[branch_ind]
    # print(curr_label)
    del (labels[branch_ind])
    # print('depth: %d' % depth)
    value_set = set([instance[branch_ind] for instance in data])
    value_set = list(value_set)

    # pdb.set_trace()
    new_labels = labels[:]
    node.left = Node(curr_label, value_set[0])
    new_data = splitData(data, branch_ind, value_set[0])
    growTree(node.left, new_data, new_labels, class_list, tree_depth + 1, max_depth)

    if len(value_set) > 1:
        new_labels = labels[:]
        node.right = Node(curr_label, value_set[1])
        new_data = splitData(data, branch_ind, value_set[1])
        growTree(node.right, new_data, new_labels, class_list, tree_depth + 1, max_depth)


def prediction(tree, instance, label_list):
    # each instance is a list [arrtri1, attri2,..., attriN]
    if tree.left == None:
        return tree.classify()
    attribute = tree.left.label
    ind = label_list.index(attribute)
    if instance[ind] == tree.left.val:
        if tree.left.left:
            return prediction(tree.left, instance, label_list)
        else:
            return tree.left.classify()
    if (tree.right) and (instance[ind] == tree.right.val):
        if tree.right.left:
            return prediction(tree.right, instance, label_list)
        else:
            return tree.right.classify()
    return "error"


# Main
if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

csv.register_dialect('mydialect',delimiter='\t',quoting=csv.QUOTE_ALL)

# Read input file
train_data = []
with open(train_input, 'r') as fin:
    csv_reader = csv.reader(fin, 'mydialect')
    labels = next(csv_reader)
    for row in csv_reader:
        train_data.append(row)
class_list = [instance[-1] for instance in train_data]
label_list = list(labels)


test_data = []
with open(test_input, 'r') as fin:
    csv_reader = csv.reader(fin, 'mydialect')
    next(csv_reader)
    for row in csv_reader:
        test_data.append(row)

root = Node(None, None)

growTree(root, train_data, labels, class_list, 0, max_depth)
print("decision tree:")
root.printNode(0)

# Output predicted labels and error rate
# train out
with open(train_out, "w") as fout:
    train_error = 0
    # print("train.labels: ")
    for instance in train_data:
        predict = prediction(root, instance, label_list)
        if predict != instance[-1]:
            train_error += 1.0
        # print(predict)
        print(predict, file=fout)
# test out
with open(test_out, "w") as fout:
    test_error = 0
    # print("test.labels: ")
    for instance in test_data:
        predict = prediction(root, instance, label_list)
        if predict != instance[-1]:
            test_error += 1.0
        # print(predict)
        print(predict, file=fout)

# metrics out
with open(metrics_out, "w") as fout:
    print("error(train): ", end='')
    print("error(train): ", end='', file=fout)
    print(train_error / len(train_data))
    print(train_error / len(train_data), file=fout)
    print("error(test): ", end='')
    print("error(test): ", end='', file=fout)
    print(test_error / len(test_data))
    print(test_error / len(test_data), file=fout)