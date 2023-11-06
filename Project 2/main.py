import __init__
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import toolbox_02450
import regression_part_a
import regression_part_b 
import classification
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

X_org = X.astype(float) 

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

regression_part_a.run(X_org, df, np, pd, plt)

#####################
#                   #
# Regression part b #
#                   #
#####################

regression_part_b.run(X_org, df, np, pd, plt)

#####################
#                   #
#  Classification   #
#                   #
#####################

classification.run(X_org, df, classLabels, np, pd, plt)

if SHOW_PLOTS:
    plt.show()