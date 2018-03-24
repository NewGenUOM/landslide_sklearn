import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


dfinput = pd.read_csv('input_data.csv')
dfoutput = pd.read_csv('outputs.csv')


dfinput = dfinput[['overburden','slope','landform','landuse']]


X = np.array(dfinput)
y = np.array(dfoutput)
#
# scaler = MinMaxScaler()  # Default behavior is to scale to [0,1]
# X = scaler.fit_transform(X)
#

X_train, X_test, y_train,y_test = train_test_split(X,y,random_state=0)

totalPointsInTrainSet = (y_train == 1).sum()+ (y_train == 0).sum()
totalPointsInTestSet = (y_test == 1).sum()+ (y_test == 0).sum()

print("Total train sets is %s and it includes %s landslide points" % (totalPointsInTrainSet, (y_train == 1).sum()))
print("Total test sets is %s and it includes %s landslide points" % (totalPointsInTestSet, (y_test == 1).sum()))


clf = SVC()


print("Started training model...")
clf.fit(X_train,y_train)
print("Train complete!")

list1 = clf.predict(X_test)
list2 = clf.predict(X_train)

print("Successfull prediction from test set %s " % ((list1 == 1).sum()))
print("Successfull prediction from train set %s " % ((list2 == 1).sum()))


accuracy = clf.score(X_test,y_test)

print("overall accuracy %s " %(accuracy))
