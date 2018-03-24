import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from project_enum import Algorithms

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

mlp = MLPClassifier(activation="relu", random_state=1, solver="adam", hidden_layer_sizes=(10, 12, 10))
clf= AdaBoostClassifier(n_estimators=100)

algorithm = int(input("Please enter the training algorithm\n 1.MLP neural nets\n 2.Adaboost\n 3.Gussian Naive Bayes\n 4.SVM\n "))

if algorithm==Algorithms.MLP_Neural_Networks:
    print("Started training model...")
    mlp.fit(x_train, y_train.ix[:, 0])
    finalScore=mlp.score(x_train,y_train.ix[:,0])
    print("Train complete!")
    print (finalScore)
    pred = mlp.predict(x_test)
    finalizePrediction(pred);
    acc=accuracy_score(y_test,pred)
    print(acc)

elif algorithm==Algorithms.AdaBoost:
    print("started training using adaBoost")
    clf.fit(x_train,y_train.ix[:,0])
    finalScore=clf.score(x_train,y_train.ix[:,0])
    print ("Ada boost train complete")
    print(finalScore);

    adaBoostprediction=clf.predict(x_test);
    finalizePrediction(adaBoostprediction);



# elif algorithm==Algorithms.NaiveBayes:
#
#     gnb= GaussianNB()
#     pred= gnb.fit(x_train,y_train.ix[:,0]).predict(x_test)
#     finalizePrediction(pred)

else:
    print("sorry")





