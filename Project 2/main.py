import __init__
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# If set to true, then plots will be shown. Scripts stops after showing the plot, thats when
# this variable may come in handy.
SHOW_PLOTS = False

#################################
#                               #
#          Project 2            #
#                               #
#################################


#####################
#                   #
#    Data import    #
#                   #
#####################

print("\n##### DATA IMPORT #####\n")

# Read the data file line by line
with open('Data/ecoli.data', 'r') as file:
    lines = file.readlines()

# Split each line into values and store them in a list of lists
data = [line.split() for line in lines]

# Create a DataFrame from the list of lists
df = pd.DataFrame(data)

df.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']

print(df.head())

cols = range(1,8)

# Extract the attributes into a matrix
X = np.asarray(df.values[:, cols])

X = X.astype(float) 

# Extract the attribute names that came from the header
attributeNames = np.asarray(df.columns[cols])

# Extracting the strings for each sample from the raw data loaded from the csv:
classLabels = df.values[:,-1] # -1 takes the last column
# Then determine which classes are in the data by finding the set of 
# unique class labels 
classNames = np.unique(classLabels)
# Assign each type of the classes with a number by making a
# Python dictionary
classDict = dict(zip(classNames,range(len(classNames))))

# Class index vector y
y = np.array([classDict[cl] for cl in classLabels])

# Determine the number of data objects and number of attributes
N, M = X.shape

# Number of classes
C = len(classNames)

#####################
#                   #
# Regression part a #
#                   #
#####################

# Check X matrix stats before feature transformation
for idx, name in enumerate(df.columns[1:-1]): # These are the attributes names.
    x = X[:,idx] # Take the whole column (all data from certain feature)
    mean_x = x.mean()
    std_x = x.std(ddof=1)
    median_x = np.median(x)
    range_x = x.max()-x.min()
    
    # print(f"Statistics for attribute: {name}")
    # print('\tMean:',mean_x)
    # print('\tStandard Deviation:',std_x)
    # print('\tMedian:',median_x)
    # print('\tRange:',range_x)

scaler = StandardScaler()
X = scaler.fit_transform(X)

# print("\n\n ############# \n \
# AFTER SCALING \n \
# ############# \n \n")

# Check X matrix stats after feature transformation
for idx, name in enumerate(df.columns[1:-1]): # These are the attributes names.
    x = X[:,idx] # Take the whole column (all data from certain feature)
    mean_x = x.mean()
    std_x = x.std(ddof=1)
    median_x = np.median(x)
    range_x = x.max()-x.min()
    
    # print(f"Statistics for attribute: {name}")
    # print('\tMean:',mean_x)
    # print('\tStandard Deviation:',std_x)
    # print('\tMedian:',median_x)
    # print('\tRange:',range_x)

#####################
#                   #
# Regression part b #
#                   #
#####################

#####################
#                   #
#  Classification   #
#                   #
#####################
print("\n##### CLASSIFICATION #####\n")
# Split dataset into 70% train og 30% test set
data_train, data_test, label_train, label_test = train_test_split(X, classLabels, test_size=.30)

## BASELINE ##

# Create a dummy classifier that predicts the most frequent class (majority class)
dummy_classifier = DummyClassifier(strategy="most_frequent")

# Fit the model on the training data and labels
dummy_classifier.fit(data_train, label_train)

# Make predictions on the test data
dummy_predictions = dummy_classifier.predict(data_test)

# Calculate accuracy to evaluate the baseline model
baseline_accuracy = accuracy_score(label_test, dummy_predictions)

print("Baseline Model Accuracy: {:.2f}%".format(baseline_accuracy * 100))

## LOGISTIC REGRESSION ##
# Standardize the training and set set based on training set mean and std
mu = np.mean(data_train, 0)
sigma = np.std(data_train, 0)
data_train = (data_train - mu) / sigma
data_test = (data_test - mu) / sigma

# Fit multinomial logistic regression model
regularization_strength = 1e-3
logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, random_state=1,
                            penalty='l2', C=1/regularization_strength, max_iter=10^1000)
logreg.fit(data_train,label_train)
data_test_est = logreg.predict(data_test)
test_error_rate = np.sum(data_test_est!=label_test) / len(label_test)

predict = lambda x: np.argmax(logreg.predict_proba(x),1)
plt.figure(2,figsize=(9,9))
toolbox_02450.visualize_decision_boundary(predict, [data_train, data_test], [label_train, label_test], attributeNames, classNames)
plt.title('LogReg decision boundaries')
plt.show()


# Number of miss-classifications
print('Error rate: \n\t {0} % out of {1}'.format(test_error_rate*100,len(label_test)))

# Calculate accuracy to evaluate the baseline model
logreg_accuracy = accuracy_score(label_test, data_test_est)

print("logreg Model Accuracy: {:.2f}%".format(logreg_accuracy * 100))

plt.figure(2, figsize=(9,9))
plt.hist([label_train, label_test, data_test_est], color=['red','green','blue'], density=True)
plt.legend(['Training labels','Test labels','Estimated test labels'])

if SHOW_PLOTS:
    plt.show()