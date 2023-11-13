from sklearn import model_selection

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, DistanceMetric
from sklearn.preprocessing import StandardScaler

import numpy as np



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
    print("\n##### CLASSIFICATION #####\n")
    
    print("\nTransforming X to make the problem binary\n")
    
    # Make the data copy to not affect original dataset 
    X_bin = X
    
    # Make the mean = 0, std, dev = 1
    scaler = StandardScaler()
    X_bin = scaler.fit_transform(X_bin)
    
    # Transform labels to make the problem binary
    #print(classLabels)
    classLabels = [1 if label == "cp" else 0 for label in classLabels]
    
    # Convert list to the np array for future calculations.
    classLabels = np.array(classLabels)
    #print(classLabels)
    
    
    # Split dataset into 70% train og 30% test set
    # data_train, data_test, label_train, label_test = model_selection.train_test_split(X_bin, classLabels, test_size=.30)

    # Use K = 10 cross-validation
    K = 10
    CV = model_selection.KFold(K, shuffle=True)
    

    Error_train = np.empty((K,1))
    Error_test = np.empty((K,1))

    dummy_err_avg = 0
    lin_reg_err_avg = 0
    knn_err_avg = 0

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

        plt.figure(2, figsize=(9,9))
        plt.hist([label_train, label_test, dummy_y_est], color=['red','green','blue'], density=True)
        plt.legend(['Training labels','Test labels','Estimated test labels'])



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

        plt.figure(2, figsize=(9,9))
        plt.hist([label_train, label_test, linear_reg_y_est], color=['red','green','blue'], density=True)
        plt.legend(['Training labels','Test labels','Estimated test labels'])



        ###########################
        ########### KNN ###########
        ###########################

        #K_neigh = [n for n in range(1, 16)]
        best_K_neigh = 13

        # Distance metric (corresponds to 2nd norm, euclidean distance).
        # You can set dist=1 to obtain manhattan distance (cityblock distance).
        dist=2
        metric = 'minkowski'
        metric_params = {} # no parameters needed for minkowski

        knn_classifier, knn_num = get_best_knn_classifier(data_train, label_train, internal_cross_validation, X_bin, classLabels)
        
        # knn_classifier = KNeighborsClassifier(n_neighbors=best_K_neigh, p=dist, 
        #                                       metric=metric,
        #                                       metric_params=metric_params)

        knn_classifier.fit(data_train, label_train)

        knn_y_est = knn_classifier.predict(data_test)
        test_error_rate_knn = np.sum(knn_y_est!=label_test) / len(label_test)


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
        

        fold_no += 1
        
    dummy_err_avg /= K
    lin_reg_err_avg /= K
    knn_err_avg /= K

    print(f'GEN. ERR\t| {dummy_err_avg*100:.2f}\t\t|\t\t{lin_reg_err_avg*100:.2f}\t\t|\t\t\t{knn_err_avg*100:.2f}')