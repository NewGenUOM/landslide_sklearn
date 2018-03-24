from sklearn.ensemble import AdaBoostClassifier

class AdaBoost:

    clf= AdaBoostClassifier(n_estimators=100)

    def train_model(self, input_training_data_set, output_training_data_set):
        print("training using Adaboost Classifier.....")
        self.clf.fit(input_training_data_set, output_training_data_set)
        print("successfully trained")


    # @staticmethod
    def predict(self, input_test_data_set):
        pred = self.clf.predict(input_test_data_set)
        return pred
