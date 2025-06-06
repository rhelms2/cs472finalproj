#
#
# Holds logic for k-nearest neightbor model
#
#
import sys
import DataHandler
import ModelTester


class NearestNeighbor():

    def __init__(self, training, k):
        self.model = training
        self.len_data = len(training)
        self.k = k

    # Squared Euclidean distance
    def __sq_distance(self, ex1, ex2):
        num_params = len(ex1)
        distance = 0
        for i in range(num_params):
            distance += (ex1[i] - ex2[i])**2
        return distance

    # Returns the average y value of the first k of examples, where examples are tuples of (distance, y)
    def __average_ky(self, examples):
        average = 0
        for i in range(self.k):
            average += examples[i][1]
        average = average / self.k
        return average

    # IMPORTANT - distances is encoded as a list of tuples, where tuple[0] = distance, and tuple[1] is the y value for that associated distance
    def predict(self, ex_to_predict):
        # Iterate over data and collect distances of each example from the example to predict
        distances = []
        for i in range(self.len_data):
            distances.append((self.__sq_distance(self.model[i][0], ex_to_predict), self.model[i][1]))
        distances = sorted(distances) # sort tuples in ascending order based off distance measure

        prediction = distances[0][1] if self.k == 1 else self.__average_ky(distances)
        return prediction


def main(argv):
    if (len(argv) != 2):
        print('Usage: python3 k-nearestneighbor.py <csv_data> <K>')
        sys.exit(2)

    # Get input data
    raw_data, varnames = DataHandler.read_data(argv[0])
    k = int(argv[1])

    # Split
    training, test, validation = DataHandler.split_data(raw_data)

    # Create model
    model = NearestNeighbor(training, k)

    # Test and print results
    MSE = ModelTester.calculateMSE(model, test)
    print("K: ", k)
    print("Mean Squared Error: ", round(MSE, 3))
    
    # Root mean squared error quantifies error using the original data's measurement units
    print("Average error in original measurement units: ", round(MSE**(0.5), 3)) 


if __name__ == '__main__':
    main(sys.argv[1:])
