import numpy as np

# load the CSV file as a numpy matrix
dataset = np.genfromtxt("training_dataset_stream.csv",
                        delimiter=',', skip_header=1)

# Let make our label to integer i.e. stream: Science : 3, Commerce : 2 and Arts : 1

# separate the data from the target attributes
y = dataset[:,5]
X = dataset[:,0:5]

# breaking the training set for set of 10 test data to check the prediction score
np.random.seed(0)
indices = np.random.permutation(len(X))
X_train = X[indices[:-10]]
y_train = y[indices[:-10]]
X_test  = X[indices[-10:]]
y_test  = y[indices[-10:]]


from sklearn.ensemble import RandomForestClassifier

#training model using Random Forest Classification algorithm
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train, y_train)

#Printing the score of our model which is close 98% is good
print rfc.score(X_train, y_train)

#predict the test data which we got from line 17
print rfc.predict(X_test)

#try and get a predication for below academic marks in an order
# Math,Science,English,Social_Studies,Language

print rfc.predict([[36,54,67,54,56]]) #predicts 1 = Arts
print rfc.predict([[96,84,67,84,56]]) #predicts 3 = Science
print rfc.predict([[56,44,67,54,56]]) #predicts 2 = Commerce
