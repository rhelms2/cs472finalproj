#
# This file contains functions used for reading csv data into usable python data structures.
#
# Based on starter code for id3 assignment provided by Daniel Lowd, 1/25/2018
#
import sys
import re
import doctest


# Load data from a file
def read_data(filename):
    file = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = file.readline().strip()
    varnames = p.split(header)
    num_params = len(varnames)
    ex_i = 0
    for line in file:
        vals = p.split(line.strip())
        # Data formatted as tuple(list(parameter_vals), learning_val)
        data.append(([], float(vals[num_params-1])))
        # Change gender to a numerical value 0 for female, 1 for male
        vals[1] = float(1) if vals[1] == "male" else float(0)
        # Don't care about User_ID as a var so we start from 1
        for i in range(1, num_params - 1):
            data[ex_i][0].append(float(vals[i]))
        ex_i+=1

    varnames.remove('User_ID')
    return (data, varnames)


def split_data(raw_data, training_perc = .7, test_perc = .2, validation_perc = .1):
    """Returns training, test, and validation splits for data
    """
    len_data = len(raw_data)

    num_training_ex = int(len_data * training_perc)
    num_test_ex = int(len_data * test_perc)
    num_val_ex = len_data - (num_training_ex + num_test_ex)

    training_data = []
    for i in range(0, num_training_ex):
        training_data.append(raw_data[i])

    test_data = []
    for i in range(i, i + num_test_ex):
        test_data.append(raw_data[i])

    validation_data = []
    for i in range(i, i + num_val_ex):
        validation_data.append(raw_data[i])

    return (training_data, test_data, validation_data)


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 1):
        print('Usage: python3 DataHandler.py <csv_data>')
        sys.exit(2)
    raw_data, varnames = read_data(argv[0])
    training, test, validation = split_data(raw_data)

    print("Done!")


if __name__ == "__main__":
    #doctest.testmod()
    main(sys.argv[1:])
