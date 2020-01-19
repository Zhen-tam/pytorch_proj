import sys

def form_dict(listWords, dict_in, threshold):
    new_dict = {}
    dict_out = {}
    for word in listWords:
        try:
            new_key = dict_in[word]
        except KeyError:
            continue
        if new_key in new_dict:
            new_dict[new_key] += 1
        else:
            new_dict[new_key] = 1

    if threshold != 0:
        for k, v in new_dict.items():
            if v < threshold:
                dict_out[k] = 1
        return dict_out

    else:
        for k, v in new_dict.items():
            dict_out[k] = 1
        return dict_out

def parseFile(file_in, file_out, dict_in, threshold):

    with open(file_in, 'r') as infile:
        file_lines = infile.readlines()

    parsed_line = ""
    for line in file_lines:
        label, words = line.split('\t')
        y = words.split(' ')
        parsed_line += label
        new_dict = form_dict(y, dict_in, threshold)

        for k,v in new_dict.items():
            parsed_line += "\t" + str(k) + ":" + str(v)
        parsed_line += "\n"

    with open(file_out, 'w') as outfile:
        outfile.write(parsed_line)

def feature_flag(train_in, valid_in, test_in, dict_in, train_out, valid_out, test_out, threshold):

    parseFile(train_in, train_out, dict_in, threshold)
    parseFile(valid_in, valid_out, dict_in, threshold)
    parseFile(test_in, test_out, dict_in, threshold)

if __name__ == "__main__":
    """
        Read the arguments and call the function
    """

    train_infile = sys.argv[1]
    vaild_infile = sys.argv[2]
    test_infile = sys.argv[3]
    dictionary  = sys.argv[4]
    train_outfile = sys.argv[5]
    valid_outfile = sys.argv[6]
    test_outfile = sys.argv[7]
    featureflag = int(sys.argv[8])

    # form the dictionary for accessing later on
    word_dict = {}
    dict_lines = {}
    threshold = 0
    with open(dictionary, 'r') as infile:
        dict_lines = infile.readlines()

    for line in dict_lines:
        line = line.strip()
        x = line.split(' ')
        word_dict[x[0]] = x[1].strip()

    if featureflag == 2:
        threshold = 4

    feature_flag(train_infile, vaild_infile, test_infile, word_dict, train_outfile, valid_outfile, test_outfile, threshold)

