import sklearn
import pandas as pd
import numpy as np
from sklearn import linear_model
#from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=';')
#print(data.head())

data = data[["G1","G2","G3","studytime","failures","absences"]]
#print(data.head())

predict = "G3"

# X return a new dataframe without G3, Y return a new dataframe with only G3
X = np.array(data.drop([predict],1)) 
Y = np.array(data[predict])

# only save the model when current accuracy > best_accuracy, train the model 30 times
best_accuracy= 0
for _ in range(30):

    # split all the data into 90% training data set, 10% test data set
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

    # store the model into a variable named linear
    linear = linear_model.LinearRegression()

    # use x_train and y_train to find the best fit line
    linear.fit(x_train, y_train)

    # get the accuracy from x_test and y_test
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        # save the model into a pickle file in the current directory
        with open("studentmodel.pickle","wb") as f:
            pickle.dump(linear, f)

# load the modle from the pickle file into linear variable
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# a line in 5 dimensioinal space has 5 coefficients for 5 attributes
print("Coefficient:" , linear.coef_)
print("Intercept:" , linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p],data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()