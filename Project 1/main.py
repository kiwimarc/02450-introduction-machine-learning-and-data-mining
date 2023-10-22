
##############################################
#                                            #
# This code was created as part of Project 1 #
# within Introduction to Machine Learning    #
# course at the Technical University of      #
# Denmark. The following code is based on    #
# the code provided in Toolbox, which is the #
# integral part of the mentioned course.     #
#                                            #
##############################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
from scipy.io import loadmat
from scipy.stats import zscore

# import seaborn as sns

# If set to true, then plots will be shown. Scripts stops after showing the plot, thats when
# this variable may come in handy.
SHOW_PLOTS = True

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

# #######################
# # Check if there is any correlation,
# # anf if the attributes appear to be normal
# # distributed 
# #######################

# # Make subset for each class

# i = 1
# fig, axs = plt.subplots(len(classNames), len(attributeNames))
# # plt.figure(figsize=(10, 6))  # Adjust the figure size
# x_axis_limits = (0, 1.0)
# for attribute_idx, attribute_name in enumerate(attributeNames):
#     for class_idx, class_name in enumerate(classNames):
#         sns.set(style="whitegrid")
#         axs[class_idx, attribute_idx].set_title(f'Atr {attribute_name}, cls {class_name}')
#         axs[class_idx, attribute_idx].set_xlabel(class_name)
#         subset = df[df.iloc[:, -1] == class_name]
#         axs[class_idx, attribute_idx].hist(subset.iloc[:, attribute_idx], range = (0,1))
#         axs[class_idx, attribute_idx].set_xlim(0, 1)
#         # sns.histplot(subset.iloc[:, attribute_idx], bins=11, kde=False, element="step", common_norm=False,
#         #             ax=axs[class_idx, attribute_idx], label=f'Class {class_name}')
#         # xticks = [i / 11 for i in range(12)]
#         # axs[class_idx, attribute_idx].set_xticks(xticks)
#         #axs.set_xticklabels([f"{x:.2f}-{x + 1/11:.2f}" for x in xticks])
        
#     # plt.xlabel(f'Feature {i+1}')
#     # plt.ylabel('Density')
#     # plt.legend()
# plt.show()

#######################
#                     #
#    Data analysis    #
#                     #
#######################

print("\n##### PCA #####\n")

# Subtract mean value from data
Y = X - np.ones((N,1))*X.mean(axis=0)

# PCA by computing SVD of Y
U,S,Vh = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure(2)
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components')
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()

# scipy.linalg.svd returns "Vh", which is the Hermitian (transpose)
# of the vector V. So, for us to obtain the correct V, we transpose:
V = Vh.T

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)
print("|||||||||")
print(V)
print("|||||||||")

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1
plt.figure(3)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plt.plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
plt.title('Ecoli data: PCA')
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))
plt.legend(classNames)
plt.grid()

# The first 4 principals components explains more than 90 percent of
# variance.
pcs = [0,1,2,3]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b'] #['r','g','b', 'o']
bw = .2
r = np.arange(1,M+1)
plt.figure(4)
for i in pcs:
    plt.bar(r+i*bw, V[:,i], width=bw)

    # ######### Experimental part #########
    # for bar, val in enumerate(V[:,i]):
    #     plt.text(bar, round(val, 3), str(round(val, 3)), ha='center', va='bottom')

    # # Calculate square of every coefficent for each analized PCA
    # V_sq = np.dot(V[:,i], V[:,i])
    # V_sq_sum = np.sum(V_sq)
    # print(V_sq_sum)
    # ######### Experimental part #########

plt.xticks(r+bw, attributeNames)
plt.title('Ecoli: PCA Component Coefficients')
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()

print(V[:,0]) # PC1.
print(V[:,1]) # PC2.
print(Y[0,:]) # First observation in dataset.
print(f"Projection onto PC1: {V[:,0]@Y[0,:]}")    # Project obervation onto PC1.
print(f"Projection onto PC2: {V[:,1]@Y[0,:]}")    # Project obervation onto PC2.

##################################
#                                #
#    Basic summary statistics    #
#                                #
##################################

print("\n##### BASIC SUMMARY STATISTICS #####\n")

# Calculate the empirical mean, sandard deviation, median and range
# for every attribute (before substracting mean value from the data):

print("Statistics for each attribute before substracting mean value from the data.")
for idx, name in enumerate(df.columns[1:-1]): # These are the attributes names.
    x = X[:,idx] # Take the whole column (all data from certain feature)
    mean_x = x.mean()
    std_x = x.std(ddof=1)
    median_x = np.median(x)
    range_x = x.max()-x.min()

    # Determine attribute type based on criteria
    if  name is 'lip':
        attribute_type = "Binary (takes only values of 0.48 and 1)"
    elif  name is 'chg':
        attribute_type = "Binary (takes only values of 0.5 and 1)"
    elif std_x == 0:
        # If standard deviation is 0, it's a constant attribute (not useful for modeling)
        attribute_type = "Constant"
    elif range_x == 0:
        # If the range is 0, it's a discrete attribute (all values are the same)
        attribute_type = "Discrete"
    elif np.isnan(mean_x) or np.isnan(std_x) or np.isnan(median_x):
        # If there are missing values in summary statistics, note it as a data issue
        attribute_type = ">> Data Issue: Missing Values <<"
    elif std_x < 1:
        # If the standard deviation is very small, it's likely a discrete attribute
        attribute_type = "Discrete"
    elif np.abs(mean_x - median_x) < 1e-5:
        # If mean and median are very close, it's likely a discrete attribute
        attribute_type = "Discrete"
    else:
        # By default, consider attribite as a continuous attribute
        attribute_type = "Continuous"

   

    # Display results
    print(f"Statistics for attribute: {name}")
    print(f"\tType: {attribute_type}")
    print('\tMean:',mean_x)
    print('\tStandard Deviation:',std_x)
    print('\tMedian:',median_x)
    print('\tRange:',range_x)
    
    
########################
#                      #
#  Data visualization  #
#                      #
########################

# boxplot every attribute
plt.figure(5)
plt.title('Ecoli: Boxplot')
plt.boxplot(Y)
plt.xticks(range(1,M+1), attributeNames, rotation=45)

# Histogram of the "lip" and "chg" attributes (suspected outliers)
plt.figure(6)
m = [2, 3]
for i in range(len(m)):
    plt.subplot(1,len(m),i+1)
    plt.hist(X[:,m[i]],50)
    plt.xlabel(attributeNames[m[i]])
    plt.xlim(0, 1)
    plt.ylim(0, N) # Make the y-axes equal for improved readability
    if i>0: plt.yticks([])
    if i==0: plt.title('Ecoli: suspected outliers')



u = np.floor(np.sqrt(M))
v = np.ceil(float(M)/u)
j = 0
plt.figure(7)
for i in [0,1,4,5,6]:
    plt.subplot(1, 5, j+1)
    plt.hist(X[:,i])
    plt.xlabel(attributeNames[i])
    plt.ylim(0, 120) # Make the y-axes equal for improved readability
    plt.xlim(0, 1)
    if i != 0: plt.yticks([])
    if i == 4: plt.title('Ecoli: Histogram (without binary attributes)')
    j = j + 1


# Variables correlation

Attributes = [0,1,4,5,6]
NumAtr = len(Attributes)

plt.figure(figsize=(12,12))
for m1 in range(NumAtr):
    for m2 in range(NumAtr):
        plt.subplot(NumAtr, NumAtr, m1*NumAtr + m2 + 1)
        for c in range(C):
            class_mask = (y==c)
            plt.plot(X[class_mask,Attributes[m2]], X[class_mask,Attributes[m1]], '.')
            if m1==NumAtr-1:
                plt.xlabel(attributeNames[Attributes[m2]])
            else:
                plt.xticks([])
            if m2==0:
                plt.ylabel(attributeNames[Attributes[m1]])
            else:
                plt.yticks([])
            if m2 == 2 and m1 == 0:
                plt.title('Ecoli: Correlation between attributes (without binary ones)')
            plt.ylim(0,1)
            plt.xlim(0,1)
plt.legend(classNames)


if SHOW_PLOTS:
    plt.show()





