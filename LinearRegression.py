#
# CIS 472/572 - Logistic Regression Template Code
#
# Author: Daniel Lowd <lowd@cs.uoregon.edu>
# Date:   2/9/2018
#
# Please use this code as the template for your solution.
#
import sys
import re
from math import log
from math import exp
from math import sqrt

MAX_ITERS = 100

# Const for when to stop learning based off of if magnitude of gradient is <=
LEARNING_CUTOFF = 0.0001


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        data.append((x, y))
    return (data, varnames)


# Train a logistic regression model using batch gradient descent
def train_lr(data, eta, l2_reg_weight):
    numexamples = len(data)
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0

    for j in range(MAX_ITERS):

        # Init sum vars for this iteration
        loss_sum = 0
        gr_wrt_w_sum = [0.0] * numvars
        gr_wrt_b_sum = 0

        for d, y in data:
            # Get activation
            a = 0.0
            for i in range(numvars):
                a += w[i] * d[i]
            a += b
            ya = y * a

            loss_sum += log(1 + exp(-ya))
            denom = (1 + exp(ya))

            # Update gradient sums
            for i in range(numvars):
                gr_wrt_w_sum[i] -= y*d[i] / denom
            gr_wrt_b_sum -= y/denom

        reg_cost = 0
        magn = 0
        # Regularize w_gradient, get cost and magnitude
        for i in range(numvars):
            gr_wrt_w_sum[i] += l2_reg_weight * w[i]
            reg_cost += w[i]**2
            magn += gr_wrt_w_sum[i]**2

        total_loss = ((1/numexamples) * loss_sum) + (l2_reg_weight/(2*numexamples)) * reg_cost
        magn = sqrt(magn)

        if (j % 10 == 0):
            # print(f"Cost function at end of iteration {j}: {total_loss}")
            # print(f"Magnitude of w gradient vector: {magn}\n")
            pass

        # Check for convergence
        if (j != 0 and magn <= LEARNING_CUTOFF):
            break
        # update weights and bias
        for i in range(numvars):
            w[i] -= eta * gr_wrt_w_sum[i]
        b -= eta * gr_wrt_b_sum


    return (w, b)


# Predict the probability of the positive label (y=+1) given the
# attributes, x.
def predict_lr(model, x):
    (w, b) = model

    activation = 0.0
    for i in range(len(x)):
        activation += w[i]*x[i]
    activation += b

    return (1 / (1 + exp(-activation))) 


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    if (len(argv) != 5):
        print('Usage: lr.py <train> <test> <eta> <lambda> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    eta = float(argv[2])
    lam = float(argv[3])
    modelfile = argv[4]

    # Train model
    (w, b) = train_lr(train, eta, lam)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        prob = predict_lr((w, b), x)
        print(prob)
        if (prob - 0.5) * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
