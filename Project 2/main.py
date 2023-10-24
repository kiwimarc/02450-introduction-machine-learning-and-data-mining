import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler

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