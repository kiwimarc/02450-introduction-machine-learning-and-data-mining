import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import svd

# Read the data file line by line
with open('Data/ecoli.data', 'r') as file:
    lines = file.readlines()

# Split each line into values and store them in a list of lists
data = [line.split() for line in lines]

# Create a DataFrame from the list of lists
df = pd.DataFrame(data)

# Convert columns 1 to 7 to numeric (assuming they are numeric data)
df.iloc[:, 1:8] = df.iloc[:, 1:8].apply(pd.to_numeric)

df.columns = ['Sequence Name', 'mcg', 'gvh', 'lip', 'chg', 'aac', 'alm1', 'alm2', 'class']

print(df.head())
print(df.shape)
