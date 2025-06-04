#
#
# Adapted from CS472/572 class materials
# Logistic Regression template code by Daniel Lowd, 2/9/2018
#
#
import sys
import re
from math import log
from math import exp
from math import sqrt
import DataHandler

# Const for when to stop learning based off of if magnitude of gradient is <=
LEARNING_CUTOFF = 0.0001


# Cost function: least mean square regression
# Train a linear regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight, epochs):
    numexamples = len(data)
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0

    for z in range(epochs):
        # Init sum vars for this iteration
        cost_sum = 0
        gr_wrt_w_sum = [0.0] * numvars
        gr_wrt_b_sum = 0

        for d, y in data:
            # Get activation
            a = predict((w, b), d)
            
            error = (a - y)
            cost_sum += error**2

            # Update gradient sums
            for j in range(numvars):
                gr_wrt_w_sum[j] += error * d[j]
            gr_wrt_b_sum += error

        magn = 0
        # Regularize w_gradient, get cost and magnitude
        gr_wrt_b_sum /= numexamples
        for i in range(numvars):
            gr_wrt_w_sum[i] = (gr_wrt_w_sum[i]/numexamples) + l2_reg_weight * w[i]
            magn += gr_wrt_w_sum[i]**2

        total_cost = (1/numexamples) * (0.5) * cost_sum
        magn = sqrt(magn)

        if (z % 10 == 0):
            print(f"Cost function at end of iteration {z}: {round(total_cost, 3)}")
            print(f"Magnitude of w gradient vector: {round(magn, 3)}\n")
            pass

        # Check for convergence
        if (j != 0 and magn <= LEARNING_CUTOFF):
            print("Model converged. Ending training...")
            break

        # update weights and bias
        for i in range(numvars):
            w[i] -= eta * gr_wrt_w_sum[i]
        b -= eta * gr_wrt_b_sum


    return (w, b)


# Predict the value of the y (continuous) given the
# input attributes, x.
def predict(model, x):
    (w, b) = model

    activation = 0.0
    for i in range(len(x)):
        activation += w[i]*x[i]
    activation += b

    return activation 


def print_model(modelfile, model, varnames):
    (w, b) = model
    # Write model file
    f = open(modelfile, "w+")
    f.write('Bias:\n%f\n' % b)
    f.write('Weights:\n')
    for i in range(len(w)):
        f.write('%s: %f\n' % (varnames[i], w[i]))


def calculateMSE(test_data, training_data, model):
    sum_squared_error = 0
    test_print_tally = 0
    for ex in test_data:
        input_params, val = ex
        pred = predict(model, input_params)

        if test_print_tally % 1000 == 0:
            print(f"Prediction: {round(pred, 3)}, actual value in test data: {val}")
            print(f"Sum of squared error at test case {round(test_print_tally, 3)}: {sum_squared_error}")
        test_print_tally += 1

        sum_squared_error += (val - pred)**2

    print("\n")
    acc = float(sum_squared_error) / len(test_data)
    return acc


def main(argv):
    if (len(argv) != 5):
        print('Usage:')
        print('python3 LinearRegression.py <csv_data> <learning_rate> <regularizer> <num_epochs> <model>')
        sys.exit(2)
    raw_data, varnames = DataHandler.read_data(argv[0])
    eta = float(argv[1])
    l2_reg_weight = float(argv[2])
    epochs = int(argv[3])
    modelfile = argv[4]
    training, test, validation = DataHandler.split_data(raw_data)
    model = train_lr(training, eta, l2_reg_weight, epochs)
    print_model(modelfile, model, varnames)
    MSE = calculateMSE(test, training, model)
    print("Mean Squared Error: ", round(MSE, 3))
    
    # Root mean squared error quantifies error using the original data's measurement units
    print("Average error in original measurement units: ", round(MSE**(0.5), 3)) 


if __name__ == "__main__":
    main(sys.argv[1:])
