#
#
# Contains methods for testing the accuracy of ML models on 'calories.csv' dataset
# When ran by itself, it will train all 3 models and compare their accuracy and computation time
#
#
import sys
import re
from math import log
from math import exp
from math import sqrt
import DataHandler
import LinearRegression
import NearestNeighbor
import NeuralNet


def calculateMSE(model, test_data):
    sum_squared_error = 0
    test_print_tally = 0
    for ex in test_data:
        input_params, val = ex
        pred = model.predict(input_params)

        if test_print_tally % 1000 == 0:
            print(f"Prediction: {round(pred, 3)}, actual value in test data: {val}")
            print(f"Sum of squared error at test case {round(test_print_tally, 3)}: {sum_squared_error}")
        test_print_tally += 1

        sum_squared_error += (val - pred)**2

    print("")
    acc = float(sum_squared_error) / len(test_data)
    return acc


def main(argv):
    if (len(argv) != 1):
        print('Usage:')
        print('python3 ModelTester.py <csv_data>')
        sys.exit(2)
    raw_data, varnames = DataHandler.read_data(argv[0])
    training, test, validation = DataHandler.split_data(raw_data)
    MSE = calculateMSE(test, training, model)
    print("Mean Squared Error: ", round(MSE, 3))
    
    # Root mean squared error quantifies error using the original data's measurement units
    print("Average error in original measurement units: ", round(MSE**(0.5), 3)) 


if __name__ == "__main__":
    main(sys.argv[1:])
