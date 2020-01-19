import numpy as np
import sys
from collections import Counter


def form_dict(filename):
    data = {}
    index = 0
    with open(filename, "r") as fin:
        for line in fin:
            line = line.strip()
            data.update({index: line})
            index += 1
    return data


def data_parse(filename):
    word_dict = {}
    tag_dict = {}
    index = 0
    with open(filename, "r") as fin:
        for line in fin:
            if index < 10000:
                word_list = []
                tag_list = []
                word_tag = line.split(" ")
                for item in word_tag:
                    word = item.split("_")[0].strip()
                    tag = item.split("_")[1].strip()
                    word_list.append(word)
                    tag_list.append(tag)

                word_dict.update({index: word_list})
                tag_dict.update({index: tag_list})
                index += 1

            else:
                return word_dict, tag_dict

    # return word_dict, tag_dict


def find_dict_index(item, diction):
    for k, v in diction.items():
        if item == v:
            return k


def cal_pai(tag_dict, tags):

    tag_num = len(tags)
    all_tag_num = 0

    pai = np.zeros((tag_num, 1), dtype="float64")

    tag_counter = {}
    for key, value in tags.items():
        tag_counter.update({value: 1})

    for k, v in tag_dict.items():
        tag_counter[v[0]] += 1

    for k0, v0 in tag_counter.items():
        all_tag_num += v0

    for k1, v1 in tag_counter.items():
        pai[find_dict_index(k1, tags)] = v1/all_tag_num


    return pai


def cal_A(tag_dict, tags):
    tag_num = len(tags)
    A = np.zeros((tag_num, tag_num), dtype="float64")
    A1 = np.zeros_like(A, dtype="float64")
    for k, v in tag_dict.items():
        for i in range(len(v) - 1):
            p = find_dict_index(v[i], tags)
            q = find_dict_index(v[i+1], tags)
            A[p][q] += 1

    for i in range(tag_num):
        for j in range(tag_num):
            A1[i][j] = (A[i][j]+1)/(np.sum(A[i]) + tag_num)

    return A1


def cal_B(word_dict, tag_dict, tags, words):
    tag_num = len(tags)
    word_num = len(words)

    B = np.zeros((tag_num, word_num), dtype="float64")
    B1 = np.zeros_like(B, dtype="float64")

    for k, v in tag_dict.items():
        for i in range(len(v)):
            p = find_dict_index(v[i], tags)
            q = find_dict_index(word_dict[k][i], words)
            B[p][q] += 1


    for i in range(tag_num):
        for j in range(word_num):
            B1[i][j] += (B[i][j]+1)/(np.sum(B[i]) + word_num)

    return B1


def write_output(filename, data):
    with open(filename, "w") as fout:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                fout.write(str(data[i][j]) + " ")
            fout.write("\n")


def main(args):
    train_in = args[1]
    index_to_word = args[2]
    index_to_tag = args[3]
    hmmprior = args[4]
    hmmemit = args[5]
    hmmtrans = args[6]

    words = form_dict(index_to_word)
    tags = form_dict(index_to_tag)
    word_dict, tag_dict = data_parse(train_in)

    pai = cal_pai(tag_dict, tags)

    A = cal_A(tag_dict, tags)

    B = cal_B(word_dict, tag_dict, tags, words)

    write_output(hmmprior, pai)
    write_output(hmmemit, B)
    write_output(hmmtrans, A)


if __name__ == "__main__":
    main(sys.argv)