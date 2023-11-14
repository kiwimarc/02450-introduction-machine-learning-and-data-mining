import scipy.stats as st
import scipy.stats

from sklearn import model_selection

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, DistanceMetric
from sklearn.preprocessing import StandardScaler

import numpy as np



def mcnemar(y_true, yhatA, yhatB, alpha=0.05):
    # perform McNemars test
    nn = np.zeros((2,2))
    c1 = yhatA - y_true == 0
    c2 = yhatB - y_true == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat)
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1)*0.5 * (Q-1)
    q = (1-Etheta)*0.5 * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1-alpha, a=p, b=q) )

    p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    return thetahat, CI, p



def get_best_dummy_classifier(data_train, label_train, cross_validation_inn, X_bin, classLabels):
    K_inn = cross_validation_inn
    CV_inn = model_selection.KFold(K_inn, shuffle=True)
    
    Error_train_inn = np.empty((K_inn,1))
    Error_test_inn = np.empty((K_inn,1))
    
    
    for train_index, test_index in CV_inn.split(X_bin, classLabels):

        data_train = X_bin[train_index,:]
        data_test = X_bin[test_index,:]
        label_train = classLabels[train_index]
        label_test = classLabels[test_index]
        
        dummy_classifier = DummyClassifier(strategy="most_frequent")
        dummy_classifier.fit(data_train, label_train)
        dummy_predictions = dummy_classifier.predict(data_test)
        
        best_error_rate_basel = 101
        test_error_rate_basel = np.sum(dummy_predictions!=label_test) / len(label_test)

        if (test_error_rate_basel < best_error_rate_basel):
            best_dummy_classifier = dummy_classifier

    return best_dummy_classifier




def get_best_logistic_regression(data_train, label_train, cross_validation_inn, X_bin, classLabels):
    K_inn = cross_validation_inn
    CV_inn = model_selection.KFold(K_inn, shuffle=True)
    
    Error_train_inn = np.empty((K_inn,1))
    Error_test_inn = np.empty((K_inn,1))
    
    
    for train_index, test_index in CV_inn.split(X_bin, classLabels):

        data_train = X_bin[train_index,:]
        data_test = X_bin[test_index,:]
        label_train = classLabels[train_index]
        label_test = classLabels[test_index]

        # Standardize the training and set set based on training set mean and std
        mu = np.mean(data_train, 0)
        sigma = np.std(data_train, 0)
        data_train = (data_train - mu) / sigma
        data_test = (data_test - mu) / sigma

        # Fit multinomial logistic regression model
        test_error_rate_regr = 101.0
        lambda_val = 1001.0
        
        # Values of lambda
        lambdas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000]

        for regularization_strength in lambdas:
            #regularization_strength = 1e-3
            logreg = LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, random_state=1,
                                        penalty='l2', C=1/regularization_strength, max_iter=10^1000)
            logreg.fit(data_train,label_train)
            data_test_est = logreg.predict(data_test)
            test_err = np.sum(data_test_est!=label_test) / len(label_test)
            if (test_err <= test_error_rate_regr):
                test_error_rate_regr = test_err
                lambda_val = regularization_strength
                best_logistic_regression = logreg
            #print(f'regularization_strength -> {regularization_strength:.5f}\ttest_err->{test_err:.5f}\ttest_error_rate_regr->{test_error_rate_regr:.5f}')
        #     print(f'{regularization_strength}\t| {test_err*100:.4f}%')
        # print('------------------------')

    return best_logistic_regression, lambda_val



def get_best_knn_classifier(data_train, label_train, cross_validation_inn, X_bin, classLabels):
    K_inn = cross_validation_inn
    CV_inn = model_selection.KFold(K_inn, shuffle=True)
    
    Error_train_inn = np.empty((K_inn,1))
    Error_test_inn = np.empty((K_inn,1))
    
    for train_index, test_index in CV_inn.split(X_bin, classLabels):

        data_train = X_bin[train_index,:]
        data_test = X_bin[test_index,:]
        label_train = classLabels[train_index]
        label_test = classLabels[test_index]
        
        test_error_rate_knn = 101.0
        best_knn_val = 201
        
        # Neighbours numbers
        K_neigh = [n for n in range(1, 30)]
        
        # Distance metric (corresponds to 2nd norm, euclidean distance).
        # You can set dist=1 to obtain manhattan distance (cityblock distance).
        dist=2
        metric = 'minkowski'
        metric_params = {} # no parameters needed for minkowski
    
        for K in K_neigh:
            knn_classifier = KNeighborsClassifier(n_neighbors=K, p=dist, 
                                                  metric=metric,
                                                  metric_params=metric_params)
            knn_classifier.fit(data_train, label_train)
            data_test_est = knn_classifier.predict(data_test)
            test_err = np.sum(data_test_est!=label_test) / len(label_test)
            if test_err < test_error_rate_knn:
                test_error_rate_knn = test_err
                best_knn_val = K
                best_knn_classifier = knn_classifier

    return best_knn_classifier, best_knn_val




def run(X, df, classLabels, np, pd, plt):
    print("\n\n###################################")
    print("##### START OF CLASSIFICATION #####")
    print("###################################\n")
    
    print("\nTransforming X to make the problem binary\n")
    
    # Make the data copy to not affect original dataset 

    #cols = [1,2,5,6,7]
    #X_bin = np.asarray(df.values[:, cols])
    cols = [0,1,4,5,6]
    X_bin = X[:, cols]
    X_bin = X_bin.astype(float)
    
    # print(X_bin)
    # print("-----------")
    # print(X)

    # Make the mean = 0, std, dev = 1
    scaler = StandardScaler()
    X_bin = scaler.fit_transform(X_bin)
    
    # Transform labels to make the problem binary
    #print(classLabels)
    classLabels = [1 if label == "im" else 0 for label in classLabels]
    
    # Convert list to the np array for future calculations.
    classLabels = np.array(classLabels)
    #print(classLabels)

    # Use K = 10 cross-validation
    K = 10
    CV = model_selection.KFold(K, shuffle=True)
    

    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))

    dummy_err_avg = 0
    lin_reg_err_avg = 0
    knn_err_avg = 0
    
    mcnemar_yhat_baseline = []
    mcnemar_yhat_linreg = []
    mcnemar_yhat_knn = []
    mcnemar_y_test = []
    
    yhat_linreg_lambd_10 = []

    avg_weights = np.zeros(5)
    avg_intercept = np.zeros(1)

    fold_no = 1
    print("################ Error rate ################")
    print("Fold\t| Baseline\t| Lambda < \tRegresion\t| KNN num\tKNN\t| Out of")
    for train_index, test_index in CV.split(X_bin, classLabels):

        data_train = X_bin[train_index,:]
        data_test = X_bin[test_index,:]
        label_train = classLabels[train_index]
        label_test = classLabels[test_index]
        internal_cross_validation = 10


        ################################
        ########### BASELINE ###########
        ################################
        
        dummy_classifier = get_best_dummy_classifier(data_train, label_train, internal_cross_validation, X_bin, classLabels)

        # Make predictions on the test data
        dummy_y_est = dummy_classifier.predict(data_test)
        test_error_rate_basel = np.sum(dummy_y_est!=label_test) / len(label_test)
        # Number of miss-classifications
        #print('Error rate: {0} % out of {1}'.format(test_error_rate_basel*100,len(label_test)))

        # Calculate accuracy to evaluate the baseline model
        baseline_accuracy = accuracy_score(label_test, dummy_y_est)
        
        #print("Baseline Model Accuracy: {:.2f}%".format(baseline_accuracy * 100))

        mcnemar_yhat_baseline.append(dummy_y_est)

        ###########################################
        ########### LOGISTIC REGRESSION ###########
        ###########################################
    
        # Standardize the training and set set based on training set mean and std
        mu = np.mean(data_train, 0)
        sigma = np.std(data_train, 0)
        data_train = (data_train - mu) / sigma
        data_test = (data_test - mu) / sigma
        
        logreg, lambda_lr = get_best_logistic_regression(data_train, label_train, internal_cross_validation, X_bin, classLabels)
        
        logreg.fit(data_train, label_train)

        linear_reg_y_est = logreg.predict(data_test)
        test_error_rate_regr = np.sum(linear_reg_y_est!=label_test) / len(label_test)

        # Number of miss-classifications
        #print('Error rate: {0} % out of {1}'.format(test_error_rate_regr*100,len(label_test)))

        # Calculate accuracy to evaluate the baseline model
        # logreg_accuracy = accuracy_score(label_test, linear_reg_y_est)
        #print("Logistic regression Model Accuracy: {:.2f}%".format(logreg_accuracy * 100))

        mcnemar_yhat_linreg.append(linear_reg_y_est)
        
        
        logreg_lambda_10 = LogisticRegression(solver='lbfgs', multi_class='multinomial', tol=1e-4, random_state=1,
                                              penalty='l2', C=1/10, max_iter=10^1000)
        logreg_lambda_10.fit(data_train,label_train)

        linear_reg_lambd_10_y_est = logreg_lambda_10.predict(data_test)
        yhat_linreg_lambd_10.append(linear_reg_lambd_10_y_est)
        #test_error_rate_regr_lambd_10 = np.sum(linear_reg_lambd_10_y_est!=label_test) / len(label_test)

        # Get the learned coefficients (weights)
        weights = logreg_lambda_10.coef_
        weights = np.concatenate(weights)
        avg_weights = avg_weights * ((fold_no - 1) / (fold_no)) + weights / (fold_no)
        #print(f"\n\nCOEFFICIENTS:\n{weights}\n AVG: {avg_weights}\n")

        intercept = logreg_lambda_10.intercept_
        intercept = np.array(intercept)
        avg_intercept = avg_intercept * ((fold_no - 1) / (fold_no)) + intercept / (fold_no)
        #print(f"\n\nINTERCEPT:\n{intercept} AVG: {avg_intercept}\n")
        
        ###########################
        ########### KNN ###########
        ###########################

        #K_neigh = [n for n in range(1, 16)]
        best_K_neigh = 13

        # Distance metric (corresponds to 2nd norm, euclidean distance).
        # You can set dist=1 to obtain manhattan distance (cityblock distance).
        # dist=2
        # metric = 'minkowski'
        # metric_params = {} # no parameters needed for minkowski

        knn_classifier, knn_num = get_best_knn_classifier(data_train, label_train, internal_cross_validation, X_bin, classLabels)
        
        # knn_classifier = KNeighborsClassifier(n_neighbors=best_K_neigh, p=dist, 
        #                                       metric=metric,
        #                                       metric_params=metric_params)

        knn_classifier.fit(data_train, label_train)

        knn_y_est = knn_classifier.predict(data_test)
        test_error_rate_knn = np.sum(knn_y_est!=label_test) / len(label_test)

        mcnemar_yhat_knn.append(knn_y_est)

        # # Compute and plot confusion matrix
        # cm = confusion_matrix(label_test, knn_y_est)
        # accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy
        # plt.figure(2)
        # plt.imshow(cm, cmap='binary', interpolation='None')
        # plt.colorbar()
        # plt.xticks(range(C)); plt.yticks(range(C))
        # plt.xlabel('Predicted class'); plt.ylabel('Actual class')
        # plt.title('Confusion matrix (Accuracy: {0}%, Error Rate: {1}%)'.format(accuracy, error_rate))

        dummy_err_avg += test_error_rate_basel
        lin_reg_err_avg += test_error_rate_regr
        knn_err_avg += test_error_rate_knn

        print(f'{fold_no}\t| {test_error_rate_basel*100:.2f}%\t| {lambda_lr:.5f}\t{test_error_rate_regr*100:.2f}%\t\t| {knn_num}\t\t{test_error_rate_knn*100:.2f}%\t| {len(label_test)}')
        
        mcnemar_y_test.append(label_test)
        fold_no += 1
        
    dummy_err_avg /= K
    lin_reg_err_avg /= K
    knn_err_avg /= K

    print(f'GEN. ERR\t| {dummy_err_avg*100:.2f}\t\t|\t\t{lin_reg_err_avg*100:.2f}\t\t|\t\t\t{knn_err_avg*100:.2f}\n')
    
    print("##################################\n")
    print("\n\n########## MCNEMAR TEST ##########\n")
    mcnemar_y_test = np.concatenate(mcnemar_y_test)
    mcnemar_yhat_baseline = np.concatenate(mcnemar_yhat_baseline)
    mcnemar_yhat_linreg = np.concatenate(mcnemar_yhat_linreg)
    mcnemar_yhat_knn = np.concatenate(mcnemar_yhat_knn)

    print("\n-----------------")
    print("Baseline vs linreg\n")
    alpha = 0.05
    [thetahat, CI, p] = mcnemar(mcnemar_y_test, mcnemar_yhat_baseline, mcnemar_yhat_linreg, alpha=alpha)

    print("\n-----------------")
    print("Baseline vs KNN\n")
    alpha = 0.05
    [thetahat, CI, p] = mcnemar(mcnemar_y_test, mcnemar_yhat_baseline, mcnemar_yhat_knn, alpha=alpha)

    print("\n-----------------")
    print("KNN vs linreg\n")
    alpha = 0.05
    [thetahat, CI, p] = mcnemar(mcnemar_y_test, mcnemar_yhat_knn, mcnemar_yhat_linreg, alpha=alpha)
    
    
    
    print(f"\n\nAVG. COEFFICIENTS: {avg_weights}")
    print(f"\n\nAVG. INTERCEPT: {avg_intercept}")
    
    
    print(f"Printing confussion matrix...")
    yhat_linreg_lambd_10 = np.concatenate(yhat_linreg_lambd_10)
    cm = confusion_matrix(yhat_linreg_lambd_10, mcnemar_y_test)
    accuracy = 100*cm.diagonal().sum()/cm.sum(); error_rate = 100-accuracy;
    cax = plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=24, color='black')
    #plt.figure(2);
    #plt.imshow(cm, cmap='binary', interpolation='None');
    # plt.colorbar()
    plt.xticks(range(2)); plt.yticks(range(2));
    plt.xlabel('Predicted class'); plt.ylabel('Actual class');
    plt.title(f"Confusion matrix (Accuracy: {accuracy:.2f}%, Error Rate: {error_rate:.2f}%");
    
    
    print("\n\n#################################")
    print("##### END OF CLASSIFICATION #####")
    print("#################################\n")


