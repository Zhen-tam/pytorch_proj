import sys
import csv
import math
import numpy as np
from collections import Counter

class Node:
    def __init__(self, attribute, val, depth):
        self.left = None
        self.right = None
        self.val = val
        self.attribute = attribute
        self.depth = depth
        self.dict1 = {}

    #form a dictionary for all data
    def form_dict(self, data):
        labels_list_tmp = []
        for item in data:
            labels_list_tmp.append(item[-1])
            self.dict1 = dict(Counter(labels_list_tmp))

    # majority vote at each node
    def classify(self):
        dist = dict(self.dist)
        k = list(dist.keys())
        v = list(dist.values())
        # print(k[v.index(max(v))])
        return k[v.index(max(v))]


def read_inf(infile):
    csv.register_dialect('mydialect', delimiter='\t', quoting=csv.QUOTE_ALL)
    with open(infile, "r") as inf:
        csv_r = csv.reader(inf,'mydialect')
        head = next(csv_r)  # remove first line
        data = []
        for row in csv_r:
            data.append(row)  # read data from tsv

    labels_list = []
    attributes_list = []
    for item in data:
        labels_list.append(item[-1])
        attributes_list = list(head)
        # del(attributes_list[-1])
        # print(attributes_list)
    return data, labels_list, attributes_list

def helper_entropy(data):
    labels_list_tmp = []
    entropy = 0.0
    for item in data:
        labels_list_tmp.append(item[-1])
    labels_dict = dict(Counter(labels_list_tmp))
    # print(labels_dict)
    for i in labels_dict:
        entropy -= labels_dict[i] / len(data) * math.log(labels_dict[i] / len(data), 2)
    return entropy

def data_update(data, attribute, val):
    data = np.array(data)
    data_new = data[data[:, attribute] == val, :]
    data_new = np.delete(data_new, attribute, axis = 1)
    data_new = list(data_new)
    return data_new

def choose_attribute(data):
    entropy = helper_entropy(data)
    MI_threshold = 0
    attribute = -1

    for i in range(len(data[0])-1):
        value_set = set([item[i] for item in data])
        con_entropy = 0
        for value in value_set:
            tmp_data = data_update(data, i, value)
            tmp_entropy = helper_entropy(tmp_data)
            con_entropy += float(len(tmp_data))/float(len(data)) * tmp_entropy
        if (entropy - con_entropy > MI_threshold):
            MI_threshold = entropy - con_entropy
            attribute = i

    return attribute

def grow_tree(node, data, attributes_list, depth, max_depth):
    node.form_dict(data)
    node.depth = depth
    new_attribute_list = [instance[-1] for instance in data]
    tmp_attribute_set = set(new_attribute_list)

    if((node.depth >= max_depth) or ((len(data[0]) -1) == 0) or
            (len(tmp_attribute_set) == 1)):
        return
    best_attribute = choose_attribute(data)
    if best_attribute == -1: return

    node_attribute = attributes_list[best_attribute]
    del attributes_list[best_attribute]

    new_attribute_list = attributes_list[:]
    value_set = set([instance[best_attribute] for instance in data])
    value_set = list(value_set)

    node.left = Node(node_attribute, value_set[0], depth + 1)
    data_new = data_update(data, best_attribute, value_set[0])
    grow_tree(node, data_new, new_attribute_list, depth + 1, max_depth)

    if(len(value_set) > 1):
        node.right = Node(node_attribute, value_set[1], depth + 1)
        data_new = data_update(data, best_attribute, value_set[1])
        grow_tree(node, data_new, new_attribute_list, depth + 1, max_depth)

# def classify(node):
#     dict1 = dict(node.dict1)
#     return max(dict1, key=dict1.get)


def prediction(tree, item, attributes_list):
    # each instance is a list [arrtri1, attri2,..., attriN]
    if tree.left is None:
        return tree.classify
    attribute = tree.left.attribute
    index = attributes_list.index(attribute)
    if item[index] == tree.left.val:
        if tree.left.left:
            return prediction(tree.left, item, attributes_list)
        else:
            return tree.left.classify
    if tree.right and (item[index] == tree.right.val):
        if tree.right.left:
            return prediction(tree.right, item, attributes_list)
        else:
            return tree.right.classify
    return "error"

def print_tree(Node):
    pass

def outf(outfile, data_in, attributes_list):
    with open(outfile, "w") as fileout:
        error_num = 0
        for item in data_in:
            predict = prediction(root, item, attributes_list)
            if predict != item[-1]:
                error_num += 1
            print(predict, file = fileout)
    error_rate = float(error_num / len(data_in))
    return error_rate





if __name__ == '__main__':
  train_input = sys.argv[1]
  test_input = sys.argv[2]
  max_depth = int(sys.argv[3])
  train_out = sys.argv[4]
  test_out = sys.argv[5]
  metrics_out = sys.argv[6]

  data, labels_list, attributes_list = read_inf(train_input)
  data_1, labels_list_1, attributes_list_1 = read_inf(test_input)
  root = Node(None, None, 0)
  grow_tree(root, data, attributes_list, 0, max_depth)
  print_tree(root)
  train_error = outf(train_out, data, attributes_list)
  test_error = outf(test_out, data, attributes_list)

  with open(metrics_out, "w") as fout:
      print("error(train): ", end='')
      print("error(train): ", end='', file=fout)
      print(train_error)
      print(train_error, file=fout)
      print("error(test): ", end='')
      print("error(test): ", end='', file=fout)
      print(test_error)
      print(test_error, file=fout)








