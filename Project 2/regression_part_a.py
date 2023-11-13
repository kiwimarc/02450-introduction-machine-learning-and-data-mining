from toolbox_02450 import rlr_validate

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
import sklearn.linear_model as lm

def run(X, df, np, pd, plt):
    print("\n##### REGRESSION PART A #####\n")

    cols = [1,2,5,6,7]

    EL_TO_PREDICT = 2

    # Extract the attributes into a matrix
    X = np.asarray(df.values[:, cols])
    X = X.astype(float)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    np.set_printoptions(threshold=np.inf, linewidth=120)
    #print(X)
    y = X[:,EL_TO_PREDICT]
    X = np.delete(X, EL_TO_PREDICT, axis=1)
    N, M = X.shape
    print(y)
    # Extract the attribute names that came from the header
    attributeNames = np.asarray(df.columns[cols])
    attributeNames = attributeNames.tolist()
    attributeNames.remove('aac')

    # Add offset attribute
    X = np.concatenate((np.ones((X.shape[0],1)),X),1)
    attributeNames = [u'Offset']+attributeNames
    print(attributeNames)
    M = M+1

    # #######################################
    # #                                     #
    # # Cross validation and regularization #
    # #                                     #
    # #######################################

    # Use K = 10 cross-validation
    K = 10
    CV = model_selection.KFold(K, shuffle=True)

    # Values of lambda
    # lambdas = np.power(10.,range(-5,12))
    lambdas = 10.0**np.linspace(-1, 6, num=71)

    # Initialize variables
    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))
    Error_train_rlr = np.empty((K,1))
    Error_test_rlr = np.empty((K,1))
    Error_train_nofeatures = np.empty((K,1))
    Error_test_nofeatures = np.empty((K,1))
    w_rlr = np.empty((M,K))
    mu = np.empty((K, M-1))
    sigma = np.empty((K, M-1))
    w_noreg = np.empty((M,K))

    k=0
    for train_index, test_index in CV.split(X,y):
        
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        internal_cross_validation = 10
        
        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

        # Standardize outer fold based on training set, and save the mean and standard
        # deviations since they're part of the model (they would be needed for
        # making new predictions) - for brevity we won't always store these in the scripts
        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)
        
        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
        
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        
        # Compute mean squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
        # Compute mean squared error without regularization
        Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
        # OR ALTERNATIVELY: you can use sklearn.linear_model module for linear regression:
        #m = lm.LinearRegression().fit(X_train, y_train)
        #Error_train[k] = np.square(y_train-m.predict(X_train)).sum()/y_train.shape[0]
        #Error_test[k] = np.square(y_test-m.predict(X_test)).sum()/y_test.shape[0]

        # Display the results for the last cross-validation fold
        if k == K-1:
            figure(k, figsize=(12,8))
            subplot(1,2,1)
            semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
            xlabel('Regularization factor', fontsize = 16)
            ylabel('Mean Coefficient Values', fontsize = 16)
            grid()
            # You can choose to display the legend, but it's omitted for a cleaner 
            # plot, since there are many attributes
            legend(attributeNames[1:], loc='best')
            
            subplot(1,2,2)
            title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)),fontsize = 16)
            loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
            xlabel('Regularization factor', fontsize = 16)
            ylabel('Squared error (crossvalidation)',fontsize = 16)
            legend(['Train error','Validation error'], fontsize = 16)
            grid()
        
        # To inspect the used indices, use these print statements
        #print('Cross validation fold {0}/{1}:'.format(k+1,K))
        #print('Train indices: {0}'.format(train_index))
        #print('Test indices: {0}\n'.format(test_index))

        k+=1
        
    #show()
    # Display results
    print('Linear regression without feature selection:')
    print('- Training error: {0}'.format(Error_train.mean()))
    print('- Test error:     {0}'.format(Error_test.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
    print('Regularized linear regression:')
    print('- Training error: {0}'.format(Error_train_rlr.mean()))
    print('- Test error:     {0}'.format(Error_test_rlr.mean()))
    print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
    print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

    print('Weights in last fold:')
    for m in range(M):
        print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))
        
