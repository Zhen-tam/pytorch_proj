import sys
import csv
# import numpy as np
import math

#read file into a list "data"
def readf(infile):
    with open(infile, "r") as inf:
        csv_r = csv.reader(inf)
        head = next(csv_r)  # remove first line
        data = []
        for row in csv_r:
            data.append(row)  # read data from tsv
    return data

#process data to what we want
def data_process(data):
    data_new = []
    for value in data:
        data_new.append(value[0].split('\t'))
    return data_new

#form a dictionary to count
def form_dict(data_new):
    data_dict = {}
    for value_new in data_new:
        if value_new[-1] not in data_dict:
            data_dict[value_new[-1]] = 1.0
        else:
            data_dict[value_new[-1]] += 1.0  # form a dictionary from data
    return data_dict

#calculate what we want
def calculation(data_new, data_dict):
    entropy = 0
    error = 1
    for i in data_dict:
        entropy -= data_dict[i] / len(data_new) * math.log(data_dict[i] / len(data_new), 2)
        if data_dict[i] / len(data_new) < error:
            error = data_dict[i] / len(data_new)
    return entropy, error

#write the answer into a file
def writef(entropy, error):
    with open(outfile, "w") as outf:
        print("entropy: ", end='', file=outf)
        print(entropy, file=outf)
        print("error: ", end='', file=outf)
        print(error, file=outf)


if __name__ == "__main__":
    infile = sys.argv[1]
    outfile = sys.argv[2]
    print("The input file is: %s" % (infile))
    print("The output file is: %s" % (outfile))
    data = readf(infile)
    data_new = data_process(data)
    data_dict = form_dict(data_new)
    entropy, error = calculation(data_new, data_dict)
    writef(entropy, error)



