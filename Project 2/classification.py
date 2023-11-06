from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import sklearn.linear_model as lm

def run(X, df, classLabels, np, pd, plt):
    print("\n##### CLASSIFICATION #####\n")
    # Split dataset into 70% train og 30% test set
    data_train, data_test, label_train, label_test = train_test_split(X, classLabels, test_size=.30)

    ## BASELINE ##
    print("\n# Base line #\n")
    # Create a dummy classifier that predicts the most frequent class (majority class)
    dummy_classifier = DummyClassifier(strategy="most_frequent")

    # Fit the model on the training data and labels
    dummy_classifier.fit(data_train, label_train)

    # Make predictions on the test data
    dummy_predictions = dummy_classifier.predict(data_test)
    test_error_rate = np.sum(dummy_predictions!=label_test) / len(label_test)
    # Number of miss-classifications
    print('Error rate: {0} % out of {1}'.format(test_error_rate*100,len(label_test)))

    # Calculate accuracy to evaluate the baseline model
    baseline_accuracy = accuracy_score(label_test, dummy_predictions)
    
    print("Baseline Model Accuracy: {:.2f}%".format(baseline_accuracy * 100))

    plt.figure(2, figsize=(9,9))
    plt.hist([label_train, label_test, dummy_predictions], color=['red','green','blue'], density=True)
    plt.legend(['Training labels','Test labels','Estimated test labels'])

    ## LOGISTIC REGRESSION ##
    print("\n# Logistic regression #\n")
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

    # Number of miss-classifications
    print('Error rate: {0} % out of {1}'.format(test_error_rate*100,len(label_test)))

    # Calculate accuracy to evaluate the baseline model
    logreg_accuracy = accuracy_score(label_test, data_test_est)

    print("Logistic regression Model Accuracy: {:.2f}%".format(logreg_accuracy * 100))

    plt.figure(2, figsize=(9,9))
    plt.hist([label_train, label_test, data_test_est], color=['red','green','blue'], density=True)
    plt.legend(['Training labels','Test labels','Estimated test labels'])

    ## METHOD 2 ##
    print("\n# Method 2 #\n")
