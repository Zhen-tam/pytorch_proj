import numpy as np
import sys


def read_parameter(filename):

    data1 = []
    with open(filename, "r") as fin:
        for line in fin:
            line = line.split(" ")
            line.remove("\n")
            new_line = []
            for ele in line:
                new_line.append(float(ele))
            data1.append(new_line)
    x = len(data1)
    y = len(data1[0])

    data2 = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            data2[i][j] = data1[i][j]
    return data2


def form_dict(filename):
    data = {}
    index = 0
    with open(filename, "r") as fin:
        for line in fin:
            line = line.strip()
            data.update({index: line})
            index += 1
    return data


def find_dict_index(item, diction):
    for k, v in diction.items():
        if item == v:
            return k


def data_parse(filename):
    word_dict = {}
    tag_dict = {}
    index = 0
    with open(filename, "r") as fin:
        for line in fin:
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

    return word_dict, tag_dict


def parse_test_line(test_word_line, diction):
    parsed_line = []
    for i in range(len(test_word_line)):
        parsed_line.append(find_dict_index(test_word_line[i], diction))
    return parsed_line


def viterbi_alg(A, B, pai, tags, words, parsed_line_word, parsed_line_tag):
    tag_num = len(tags)
    word_num = len(words)
    test_word_num = len(parsed_line_word)

    former = pai

    W = np.zeros((tag_num, test_word_num))
    P = np.zeros_like(W)

    for t in range(test_word_num):
        if t == 0:
            for i in range(tag_num):
                W[i][t] = np.log(former[i]) + np.log(B[i][parsed_line_word[t]])
                P[i][t] = parsed_line_tag[t]
        else:
            for j in range(tag_num):
                W[j][t] = np.log(B[j][parsed_line_word[t]]) \
                             + np.max(np.log(A[:, j]) + W[:, (t-1)])

                P[j][t] = np.argmax(np.log(A[:, j]) + W[:, (t-1)])


    return W, P


def retrive_seq(W, P, parsed_line_tag):
    T = len(parsed_line_tag)
    seq_y = np.array([np.argmax(W[:, T-1])])

    for i in range(T-1):
        m = int(seq_y[- 1 - i])
        seq_y = np.append(P[m][T - i - 1], seq_y)

    return seq_y


def write_output(filename, parsed_line_tag, seq_y):
    with open(filename, "w") as fout:
        for i in range(len(parsed_line_tag)):
            fout.write(words[parsed_line_word[i]] + "_" + tags[seq_y[i]] + " ")
        fout.write("\n")


if __name__ == '__main__':
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predict_out = sys.argv[7]
    metrics_out = sys.argv[8]

    pai = read_parameter(hmmprior)
    B = read_parameter(hmmemit)
    A = read_parameter(hmmtrans)

    tags = form_dict(index_to_tag)
    words = form_dict(index_to_word)

    test_word, test_tag = data_parse(test_input)

    error = 0
    words_number = 0
    accuracy = 0.0

    with open (predict_out, "w") as fout:
        for i in range(len(test_word)):
            parsed_line_word = parse_test_line(test_word[i], words)
            parsed_line_tag = parse_test_line(test_tag[i], tags)
            W, P = viterbi_alg(A, B, pai, tags, words, parsed_line_word, parsed_line_tag)
            seq_y = retrive_seq(W, P, parsed_line_tag)
            words_number += len(seq_y)
            for m in range(len(parsed_line_tag)-1):
                fout.write(words[parsed_line_word[m]] + "_" + tags[seq_y[m]] + " ")
            n = len(parsed_line_tag)-1
            fout.write(words[parsed_line_word[n]] + "_" + tags[seq_y[n]])
            fout.write("\n")

            for j in range(len(seq_y)):
                if seq_y[j] != parsed_line_tag[j]:
                    error += 1
        accuracy = 1 - error / words_number

    with open(metrics_out, "w") as fo:
        fo.write("Accuracy: " + str(accuracy) + "\n")




