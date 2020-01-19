import sys
import csv
import math
import numpy as np
from collections import Counter


class Node:
    def __init__(self, label, val, depth):
        self.left = None
        self.right = None
        self.label = label
        self.val = val
        self.depth = depth
        self.diction = {}

    def form_dict(self,data):
        labels_list_tmp = []
        for item in data:
            labels_list_tmp.append(item[-1])
            self.diction = dict(Counter(labels_list_tmp))



def majority_vote(node):
    diction = dict(node.diction)
    return max(diction, key=diction.get)


def helper_entropy(data):
    labels_list_tmp = []
    entropy = 0.0
    for item in data:
        labels_list_tmp.append(item[-1])
    labels_dict = dict(Counter(labels_list_tmp))
    for i in labels_dict:
        entropy -= labels_dict[i] / len(data) * math.log(labels_dict[i] / len(data), 2)
    return entropy


def data_update(data, attribute, value):
    data = np.array(data)
    new_data = data[data[:, attribute] == value, :]
    new_data = np.delete(new_data, attribute, axis = 1)
    new_data = list(new_data)
    return new_data


def choose_branch(data):
    entropy = helper_entropy(data)
    threshold_MI = 0
    max_MI = 0
    attribute = -1

    for i in range(len(data[0]) - 1):
        value_set = set([instance[i] for instance in data])
        conditional_ent = 0
        for value in value_set:
            tmp_data = data_update(data, i, value)
            conditional_ent += float(len(tmp_data)) / float(len(data)) * helper_entropy(tmp_data)
        if entropy - conditional_ent > threshold_MI and entropy - conditional_ent > max_MI:
            max_MI = entropy - conditional_ent
            attribute = i
    return attribute


def growTree(node, data, attribute_list, class_list, tree_depth, max_depth):
    node.form_dict(data)
    node.depth = tree_depth

    new_class_list = [instance[-1] for instance in data]
    if new_class_list.count(class_list[0]) == len(new_class_list) or len(data[0]) == 1 \
            or node.depth >= max_depth:
        return

    best_branch = choose_branch(data)
    if best_branch == -1:
        return

    tmp_label = attribute_list[best_branch]
    del (attribute_list[best_branch])
    value_set = set([instance[best_branch] for instance in data])
    value_set = list(value_set)

    new_attribute = attribute_list[:]
    node.left = Node(tmp_label, value_set[0], tree_depth)
    new_data = data_update(data, best_branch, value_set[0])
    growTree(node.left, new_data, new_attribute, class_list, tree_depth + 1, max_depth)

    if len(value_set) > 1:
        new_attribute = attribute_list[:]
        node.right = Node(tmp_label, value_set[1], tree_depth)
        new_data = data_update(data, best_branch, value_set[1])
        growTree(node.right, new_data, new_attribute, class_list, tree_depth + 1, max_depth)


def prediction(tree, instance, attribute_list):
    # each instance is a list [arrtri1, attri2,..., attriN]
    if tree.left is None:
        return majority_vote(tree)
    attribute = tree.left.label
    ind = attribute_list.index(attribute)
    if instance[ind] == tree.left.val:
        if tree.left.left:
            return prediction(tree.left, instance, attribute_list)
        else:
            return majority_vote(tree.left)
    if tree.right and (instance[ind] == tree.right.val):
        if tree.right.left:
            return prediction(tree.right, instance, attribute_list)
        else:
            return majority_vote(tree.right)
    return "error"

def read_inf(infile):
    data = []
    with open(infile, 'r') as inf:
        csv_reader = csv.reader(inf, 'mydialect')
        labels = next(csv_reader)
        for row in csv_reader:
            data.append(row)
    class_list = [instance[-1] for instance in data]
    attribute_list = list(labels)
    return data, class_list, attribute_list, labels


def write_outf(outfile, data):
    error_num = 0
    with open(outfile, 'w') as outf:
        for instance in data:
            predict = prediction(root, instance, attribute_list)
            if predict != instance[-1]:
                error_num += 1
            outf.write(predict + '\n')
    error_rate = error_num / len(data)
    return error_rate


def print_tree(node, level):
    if node.left:
        print_tree(node.left, level + 1)
    if node.right:
        print_tree(node.right, level + 1)
    print("| " * level, end='\b')
    if node.label:
        print(node.label, end=" = ")
    if node.val:
        print(node.val, end=': ')
    str_diction = str(node.diction).replace('{', '[')
    str_diction = str_diction.replace('}', ']')
    str_diction = str_diction.replace(", ", "/ ")
    str_diction = str_diction.replace('\'', '')
    str_diction = str_diction.replace(':', '')
    print(str_diction)


if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]

csv.register_dialect('mydialect',delimiter='\t',quoting=csv.QUOTE_ALL)

train_data, class_list, attribute_list, labels = read_inf(train_input)
test_data, class_list1, attribute_list1, labels1 = read_inf(test_input)
print(attribute_list)

root = Node(None, None, 0)

growTree(root, train_data, labels, class_list, 0, max_depth)

train_error = write_outf(train_out, train_data)
test_error = write_outf(test_out, test_data)

# print tree
print("decision tree:")
print_tree(root, 0)

print("error(train): " + str(train_error))
print("error(test): " + str(test_error))

with open(metrics_out, "w") as fout:
    fout.write("error(train): " + str(train_error) + '\n')
    fout.write("error(test): " + str(test_error))

