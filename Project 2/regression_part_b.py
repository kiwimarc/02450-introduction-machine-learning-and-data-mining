from toolbox_02450 import rlr_validate
import toolbox_02450
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import sklearn.linear_model as lm
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_score

def run(X, df, np, pd, plt):
    print("\n##### REGRESSION PART B #####\n")


    # Define the range of hidden units
    hidden_units = [1, 5, 10, 15, 20, 25, 30, 35, 40, 80, 120]
    K=10 #no. of folds
    y = df['aac'] 

    # Assuming 'n' is the number of features in your dataset
    n = X.shape[1] 
    mu = np.empty((K, n-1))
    sigma = np.empty((K, n-1))

    CV = model_selection.KFold(K, shuffle=True)

    # Initialize variables
    Error_train_ann = np.empty((K,1))
    Error_test_ann = np.empty((K,1))
    Error_train_lr = np.empty((K,1))
    Error_test_lr = np.empty((K,1))
    Error_train_baseline = np.empty((K,1))
    Error_test_baseline = np.empty((K,1))

    k=0
    for train_index, test_index in CV.split(X,y):
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        # Standardize outer fold based on training set
        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)
        
        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
        
        # Apply ANN model
        for h in hidden_units:
            model = MLPRegressor(hidden_layer_sizes=(h,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=1000, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)
            
            # Perform internal cross-validation to find optimal number of hidden units
            cv_scores = cross_val_score(model, X_train, y_train, cv=10)
            
            # Compute mean squared error for train and test set
            Error_train_ann[k] = np.square(y_train-model.predict(X_train)).sum()/y_train.shape[0]
            Error_test_ann[k] = np.square(y_test-model.predict(X_test)).sum()/y_test.shape[0]
        
        # Apply linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Compute mean squared error for train and test set
        Error_train_lr[k] = np.square(y_train-model.predict(X_train)).sum()/y_train.shape[0]
        Error_test_lr[k] = np.square(y_test-model.predict(X_test)).sum()/y_test.shape[0]
        
        # Apply baseline model
        model = DummyRegressor(strategy="mean")
        model.fit(X_train, y_train)
        
        # Compute mean squared error for train and test set
        Error_train_baseline[k] = np.square(y_train-model.predict(X_train)).sum()/y_train.shape[0]
        Error_test_baseline[k] = np.square(y_test-model.predict(X_test)).sum()/y_test.shape[0]
        
        k += 1

    # Perform statistical comparison
    t_stat, p_val = stats.ttest_rel(Error_test_ann, Error_test_lr)
    print(f"ANN vs. Linear Regression: t-statistic = {t_stat}, p-value = {p_val}")

    t_stat, p_val = stats.ttest_rel(Error_test_ann, Error_test_baseline)
    print(f"ANN vs. Baseline: t-statistic = {t_stat}, p-value = {p_val}")

    t_stat, p_val = stats.ttest_rel(Error_test_lr, Error_test_baseline)
    print(f"Linear Regression vs. Baseline: t-statistic = {t_stat}, p-value = {p_val}")
