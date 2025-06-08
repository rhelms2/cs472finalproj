#
#
# Holds implementation of neural net to approximate a continuous function
#
#
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import DataHandler
import MSE as ModelTester

class NeuralNet(nn.Module):

    def __init__(self, training, eta, epochs, b_print):
        super().__init__()
        
        # Separate features and targets
        tr_features = [feature for feature, label in training]
        tr_targets = [target for text, target in training]

        input_dim = len(tr_features[0])

        # Perform operations on hosted GPU if possible
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        torch.device(device)

        # Encode features and targets as tensors for pytorch lib
        encoded_tr_features = torch.tensor(tr_features, dtype=torch.float32)
        encoded_tr_targets = torch.tensor(tr_targets, dtype=torch.float32).view(-1, 1)

        # Moves data from CPU to GPU
        encoded_tr_features.to(device)
        encoded_tr_targets.to(device)

        self.model = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=eta)
        self.train_nn(criterion, optimizer, epochs, encoded_tr_features, encoded_tr_targets, b_print)
        print("Finished training\n")

    def train_nn(self, criterion, optimizer, num_epochs, features, targets, b_print):
        for epoch in range(num_epochs):
            # Forward pass
            outputs = self.model(features)
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if b_print and (epoch % 10 == 0):
                print(f"Epoch {epoch} Loss: {loss.item():.4f}")

    def forward(self, x):
        return self.model(x)

    # To use ModelTester, have to convert inputs to tensors and outputs to floats
    def predict(self, x):
        return float(self.forward(torch.tensor(x, dtype=torch.float32)))


def main(argv):
    if (len(argv) != 3):
        print('Usage:')
        print('python3 NeuralNet.py <csv_data> <learning_rate> <num_epochs>')
        sys.exit(2)

    # Read args
    raw_data, varnames = DataHandler.read_data(argv[0])
    ETA = float(argv[1])
    EPOCHS = int(argv[2])

    # Split data
    training, test, validation = DataHandler.split_data(raw_data)

    # Build and train model
    model = NeuralNet(training, ETA, EPOCHS, b_print=True)

    # Test model on unseen data
    #model.eval()
    #with torch.no_grad():
        #test_loss = sum(criterion(model(torch.tensor(X, dtype=torch.float32)), torch.tensor(y, dtype=torch.float32).view(1, 1)) for X, y in test) / len(test)
    #print(f"Test Loss: {test_loss.item():.4f}")

    MSE = calculateMSE(model, test)
    print("Mean Squared Error: ", round(MSE, 3))

    # Root mean squared error quantifies error using the original data's measurement units
    print("Average error in original measurement units: ", round(MSE**(0.5), 3))


if __name__ == '__main__':
    main(sys.argv[:1])
    # main(["calories.csv", 0.0001, 10000])