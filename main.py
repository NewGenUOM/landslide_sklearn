import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from project_enum import Algorithms
from algorithms.mlp_classifier import MLPNeuralNetwork
from algorithms.adaboost import AdaBoost

def finalizePrediction(predictionArray):
    print("Predicted landslides cells:", (predictionArray == 1).sum())
    print("Actual landslides cells:", np.count_nonzero(y_test))

    predicitonLoc = np.where(predictionArray == 1)[0]  #predicted landslide locations
    actualLoc = np.where(y_test == 1)[0]    #actual landslide locations

    #counting correct predictions
    counter = 0
    for i in predicitonLoc:
        if(i in actualLoc):
            counter += 1

    print("Correct predictions: ", counter)

input_data = pd.read_csv("input_data.csv")
output_data = pd.read_csv("outputs.csv")

#divide train test sets
x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, random_state=0)

try:
    while True:

        algorithm = int(input("Please enter the training algorithm\n 1.MLP neural nets\n 2.Adaboost\n 3.Gussian Naive Bayes\n"))

        if algorithm == Algorithms.MLP_Neural_Networks:
            mlp_neural_network = MLPNeuralNetwork();
            mlp_neural_network.train_model(x_train, y_train.ix[:, 0])
            predictionArray=mlp_neural_network.mlp_predict(x_test)
            finalizePrediction(predictionArray)

        elif algorithm == Algorithms.AdaBoost:
            adaboost= AdaBoost()
            adaboost.train_model(x_train,y_train.ix[:, 0])
            predictionArray =adaboost.predict(x_test)

        else:
            print("wrong input")
            continue

except KeyboardInterrupt:
    pass

