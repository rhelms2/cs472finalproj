## Overview
This repository contains the code files submitted for my final project for CS 472: Machine Learning at University of Oregon. The goal of the project was to compare various machine learning models on fitting a simple dataset measuring calorie expenditure during workouts. The three models compared are a k-nearest neighbor model (implemented from scratch), linear regression (also implemented from scratch), and a neural network (pytorch is used here). 

## Structure
Each model is implemented in their own separate .py file. These can be run on their own and take input arguments for a dataset and to adjust the hyperparameters of the model. If a model is run on its own, it will be trained and tested and output the loss and accuracy. Each model is dependent on 2 files:

  - DataHandler.py: Contains logic to transform the calories.csv file into a usable format for each model as well as get splits for training, test and validation portions
  - MSE.py: Contains logic for finding the mean squared error of a model's performance on testing data

ModelTester.py imports all of these files and compares the training time, test time, and accuracy of each model. The hyperparamaters for each model it compares are defined in hyperparams.py.

Lastly, VisualizeData.py outputs visual graphs of calorie expenditure vs. each feature in the dataset over the whole distribution of the data.  
