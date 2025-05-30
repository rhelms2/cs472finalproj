# Holds logic for k-nearest neightbor model used to predict calorie expenditure
import sys
import DataHandler


# Squared Euclidean distance
def sq_distance(ex1, ex2):
    num_params = len(ex1)
    distance = 0
    for i in range(num_params):
        distance += (ex1[i] - ex2[i])**2
    return distance

# Returns the average y value of the first k of examples, where examples are tuples of (distance, y)
def average_ky(examples, k):
    average = 0
    for i in range(k):
        average += examples[i][1]
    average = average / k
    return average

# IMPORTANT - distances is encoded as a list of tuples, where tuple[0] = distance, and tuple[1] is the y value for that associated distance
def predict(data, ex_to_predict, k):
    len_data = len(data)
    # Iterate over data and collect distances of each example from the example to predict
    distances = []
    for i in range(len_data):
        distances.append((sq_distance(data[i][0], ex_to_predict[0]), data[i][1]))
    distances = sorted(distances) # sort tuples in ascending order based off distance measure

    prediction = distances[0][1] if k == 1 else average_ky(distances, k)
    return prediction


def calculateMSE(test_data, training_data, k):
    sum_squared_error = 0
    for ex in test_data:
        pred = predict(training_data, ex, k)
        # print(f"Prediction: {pred}, actual value in test data: {ex[1]}")

        sum_squared_error += (ex[1] - pred)**2

    acc = float(sum_squared_error) / len(test_data)
    return acc


def main(argv):
    if (len(argv) != 2):
        print('Usage: python3 k-nearestneighbor.py <csv_data> <K>')
        sys.exit(2)
    raw_data, varnames = DataHandler.read_data(argv[0])
    training, test, validation = DataHandler.split_data(raw_data)
    K = int(argv[1])
    MSE = calculateMSE(test, training, K)
    print("K: ", K)
    print("Mean Squared Error: ", round(MSE, 3))
    
    # Root mean squared error quantifies error using the original data's measurement units
    print("Average error in original measurement units: ", round(MSE**(0.5), 3)) 


if __name__ == '__main__':
    main(sys.argv[1:])