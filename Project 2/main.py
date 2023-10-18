import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd

# If set to true, then plots will be shown. Scripts stops after showing the plot, thats when
# this variable may come in handy.
SHOW_PLOTS = False

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
plt.figure(1)
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

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1
plt.figure(2)
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
plt.figure(3)
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

if SHOW_PLOTS:
    plt.show()


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

    # Display results
    print(f"Statistics for attribute: {name}")
    print('\tMean:',mean_x)
    print('\tStandard Deviation:',std_x)
    print('\tMedian:',median_x)
    print('\tRange:',range_x)
