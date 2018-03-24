
from sklearn.neural_network import MLPClassifier


class MLPNeuralNetwork:

    mlp = MLPClassifier(activation="relu", random_state=1, solver="adam", hidden_layer_sizes=(10, 12, 10))

    # @staticmethod
    def train_model(self,input_training_data_set,output_training_data_set):
        print("training using MLP Classifier.....")
        self.mlp.fit(input_training_data_set,output_training_data_set)
        print("successfully trained")


    # @staticmethod
    def predict(self,input_test_data_set):
        pred = self.mlp.predict(input_test_data_set)
        return pred


# def finalizePrediction(predictionArray):
#     print("Predicted landslides cells:", (predictionArray == 1).sum())
#     print("Actual landslides cells:", np.count_nonzero(y_test))
#
#     predicitonLoc = np.where(predictionArray == 1)[0]  #predicted landslide locations
#     actualLoc = np.where(y_test == 1)[0]    #actual landslide locations
#
#     #counting correct predictions
#     counter = 0
#     for i in predicitonLoc:
#         if(i in actualLoc):
#             counter += 1
#
#     print("Correct predictions: ", counter)
#
# input_data = pd.read_csv("input_data.csv")
# output_data = pd.read_csv("outputs.csv")
#
# #divide train test sets
# x_train, x_test, y_train, y_test = train_test_split(input_data, output_data, random_state=0)





# algorithm = int(input("Please enter the training algorithm\n 1.MLP neural nets\n 2.Adaboost\n 3.Gussian Naive Bayes\n"))

# if algorithm==Algorithms.MLP_Neural_Networks:
#     print("Started training model...")
#     mlp.fit(x_train, y_train.ix[:, 0])
#     finalScore=mlp.score(x_train,y_train.ix[:,0])
#     print("Train complete!")
#     print (finalScore)
#     pred = mlp.predict(x_test)
#     finalizePrediction(pred);
#     acc=accuracy_score(y_test,pred)
#     print(acc)

# elif algorithm==Algorithms.AdaBoost:
#     print("started training using adaBoost")
#     clf.fit(x_train,y_train.ix[:,0])
#     finalScore=clf.score(x_train,y_train.ix[:,0])
#     print ("Ada boost train complete")
#     print(finalScore);
#
#     adaBoostprediction=clf.predict(x_test);
#     finalizePrediction(adaBoostprediction);



# elif algorithm==Algorithms.NaiveBayes:
#
#     gnb= GaussianNB()
#     pred= gnb.fit(x_train,y_train.ix[:,0]).predict(x_test)
#     finalizePrediction(pred)

# else:
#     print("sorry")





