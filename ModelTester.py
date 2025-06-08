#
#
# Contains methods for testing the accuracy of ML models on 'calories.csv' dataset
# When ran by itself, it will train all 3 models and compare their accuracy and computation time
#
#
import sys
import time
import DataHandler
from LinearRegression import *
from NearestNeighbor import *
from NeuralNet import *
from hyperparams import *
from MSE import calculateMSE


MODELS = ["Nearest Neighbor", "Linear Regression", "Neural Net"]


def main(argv):
    if (len(argv) != 1):
        print('Usage:')
        print('python3 ModelTester.py <csv_data>')
        sys.exit(2)
    raw_data, varnames = DataHandler.read_data(argv[0])
    training, test, validation = DataHandler.split_data(raw_data)

    len_test = len(test)

    training_times = []
    prediction_times = []
    test_losses = []

    print("Training Nearest Neighbor:")
    tr_start1 = time.time()
    model1 = NearestNeighbor(training, K)
    tr_end1 = time.time()
    tr_time1 = tr_end1 - tr_start1
    print(f"Training time: {tr_time1} seconds")
    training_times.append(tr_time1)
    print("\nTesting Nearest Neighbor:\n")
    tst_start1 = time.time()
    MSE1 = calculateMSE(model1, test)
    tst_end1 = time.time()
    tst_time1 = tst_end1 - tst_start1
    test_losses.append(MSE1)
    prediction_times.append(tst_time1)
    print(f"Testing time for {len_test} predictions: {tst_time1} seconds")
    print(f"Average predition time: {tst_time1 / len_test} seconds")
    print("Mean Squared Error: ", round(MSE1, 3))
    print("Average squared error in original measurement units: ", round(MSE1**(0.5), 3)) 

    print(f"\nTraining Linear Regression for {NUM_EPOCHS} epochs:")
    tr_start2 = time.time()
    model2 = LinearRegression(training, LR_ETA, LR_REGULARIZER, NUM_EPOCHS, b_print=False)
    tr_end2 = time.time()
    tr_time2 = tr_end2 - tr_start2
    print(f"Training time: {tr_time2} seconds")
    training_times.append(tr_time2)
    print("\nTesting Linear Regression:\n")
    tst_start2 = time.time()
    MSE2 = calculateMSE(model2, test)
    tst_end2 = time.time()
    tst_time2 = tst_end2 - tst_start2
    test_losses.append(MSE2)
    prediction_times.append(tst_time2)
    print(f"Testing time for {len_test} predictions: {tst_time2} seconds")
    print(f"Average prediction time: {tst_time2 / len_test} seconds")
    print("Mean Squared Error: ", round(MSE2, 3))
    print("Average squared error in original measurement units: ", round(MSE2**(0.5), 3)) 
    
    print(f"\nTraining Neural Network for {NUM_EPOCHS} epochs:")
    tr_start3 = time.time()
    model3 = NeuralNet(training, NN_ETA, NUM_EPOCHS, b_print=False)
    tr_end3 = time.time()
    tr_time3 = tr_end3 - tr_start3
    print(f"Training time: {tr_time3} seconds")
    training_times.append(tr_time3)
    print("\nTesting Neural Net:\n")
    tst_start3 = time.time()
    MSE3 = calculateMSE(model3, test)
    tst_end3 = time.time()
    tst_time3 = tst_end3 - tst_start3
    test_losses.append(MSE3)
    prediction_times.append(tst_time3)
    print(f"Testing time for {len_test} predictions: {tst_time3}")
    print(f"Average prediction time: {tst_time3 / len_test}")
    print("Mean Squared Error: ", round(MSE3, 3))
    print("Average squared error in original measurement units: ", round(MSE3**(0.5), 3)) 

    overall_times = []
    for i in range(len(training_times)):
        overall_times.append(training_times[i] + prediction_times[i])

    print(f"\nBest training time: {MODELS[training_times.index(min(training_times))]} {min(training_times)}")
    print(f"Best test time: {MODELS[prediction_times.index(min(prediction_times))]} {min(prediction_times)}")
    print(f"Best overall time: {MODELS[overall_times.index(min(overall_times))]} {min(overall_times)}")
    print(f"Best accuracy: {MODELS[test_losses.index(min(test_losses))]} {min(test_losses)}")

if __name__ == "__main__":
    main(sys.argv[1:])
