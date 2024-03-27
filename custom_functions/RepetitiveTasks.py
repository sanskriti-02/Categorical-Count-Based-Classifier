 # Importing necessary libraries and packages
import time
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from imblearn.over_sampling import SMOTEN
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score,confusion_matrix,matthews_corrcoef,precision_score,recall_score,f1_score, roc_auc_score
from cleanlab.filter import find_label_issues
from sklearn.impute import KNNImputer
from sklearn.naive_bayes import CategoricalNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from CountEst.classifiers import CategoricalCBC
from custom_functions.OtherClassifiers import NaiveBayesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from prettytable import PrettyTable 
from sklearn.model_selection import train_test_split

# Dictionary containing all classifier names and their instances
classifiers = {

                'Decision Tree': DecisionTreeClassifier(),

                'K-Nearest Neighbours': KNeighborsClassifier(),

                'Support Vector Machine': SVC(probability = True),

                'Naive Bayes Classifier': CategoricalNB(),

                'Logistic Regression': LogisticRegression(max_iter = 5000),

                'Multi Layer Perceptron': MLPClassifier(max_iter = 5000),

                'AdaBoost Classifier': AdaBoostClassifier(),

                'Random Forest': RandomForestClassifier(),

                'Gradient Boosting': GradientBoostingClassifier(),
            
                'Extra Trees': ExtraTreesClassifier(),

                'XGBoost': XGBClassifier(),

                'Custom Naive Bayes Classifier': NaiveBayesClassifier(),

                'Count-Based Classifier': CategoricalCBC()
    
              }

############################################################

# Function for imputing missing values using KNNImputer
def imputation(df):
    
    # Determine the number of neighbors for KNN imputation using the square root of the dataframe's row count
    n = round(sqrt(df.shape[0]))

    # Initialize KNNImputer with the determined number of neighbors
    imputer = KNNImputer(n_neighbors=n)

    # Perform KNN imputation on the dataframe and convert the result back to a dataframe with original column names
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Round the imputed values in each column to the nearest integer
    for column in df:
        df[column] = df[column].round()

    # Return the dataframe with imputed values
    return df

############################################################

# Function for removing classes having <= 4 instances
def remove_class(df, target):

    # Count the occurrences of each class in the target column and convert to a dictionary
    class_counts = df[target].value_counts().to_dict()

    # Iterate through each class and check if it has <= 4 instances
    for key, value in class_counts.items():

        # Remove rows where the target column has the class with <= 4 instances
        if value <= 4:
            df = df[df[target] != key]

    # Return the dataframe with removed classes
    return df

############################################################

# Function for encoding the dataset
def encoding(df):
    
    # Create an instance of Ordinal Encoder
    encoder = OrdinalEncoder()
    
    # Encode categorical values using Ordinal Encoder
    df = pd.DataFrame(encoder.fit_transform(df),columns=df.columns)

    # Return the dataframe with encoded values
    return df
                                        
############################################################

# Function for separating dependent and independent variables
def separating(df, target):

    # Create a copy of the dataframe to keep the original data intact
    X = df.copy()

    # Remove the target variable from the independent variables (features)
    X.drop([target], axis=1, inplace=True)

    # Extract the target variable as the dependent variable
    y = df[target]

    # Return the independent variables (features) and the dependent variable
    return X, y

############################################################

# Function for oversampling imbalanced data
def oversampling(X, y):

    # Create an instance of SMOTEN
    resample = SMOTEN()

    # Perform oversampling on the dataset, generating synthetic samples for the minority class
    X, y = resample.fit_resample(X, y)

    # Return the oversampled independent variables (features) and dependent variable
    return X, y

############################################################
    
# Function for all classification model's training, testing and evaluation for the dataset
def classification_evaluation(x_train, x_test, y_train, y_test, models = classifiers):

    # Create an empty DataFrame to store evaluation metrics for each model
    df = pd.DataFrame(columns = ['Model','Accu','Reca','Prec','F1','AUC','MCC','TT'])

    # Create an empty list to store confusion matrices for each model
    conf = []

    # Iterate through each classifier in the predefined dictionary 'classifiers'
    for classifier in models.keys():
        
        try:

            # Determine the number of classes in the target variable
            classes = len(np.unique(y_train))

            # Get the classifier model from the 'classifiers' dictionary
            model = models[classifier]
            
            # Fitting model onto data and predicting target value for test dataset
            start_time = time.time()
            model.fit(x_train,y_train)
            y_pred = model.predict(x_test)
            y_proba = model.predict_proba(x_test)

            # Calculate AUC differently based on the number of classes
            if classes == 2:
                auc = roc_auc_score(y_test, y_pred, average='micro')
            else:
                auc = roc_auc_score(y_test, y_proba, average='macro', multi_class='ovr')

            # Store evaluation metrics in the DataFrame for each classifier
            df.loc[len(df.index)] = [classifier, accuracy_score(y_test,y_pred), 
                                     recall_score(y_test, y_pred, average='micro'), 
                                     precision_score(y_test, y_pred, average='micro'), 
                                     f1_score(y_test, y_pred, average='micro'), 
                                     auc, matthews_corrcoef(y_test,y_pred), 
                                     time.time() - start_time]

            # Store the confusion matrix for each classifier in the list
            conf.append([classifier, confusion_matrix(y_test, y_pred)])

        except Exception as e:
            
            # Print any exceptions that may occur during the process for debugging purposes
            print(classifier,':',e)
            continue
            
    # Return the DataFrame with evaluation metrics and the list of confusion matrices
    return df,conf

############################################################

# Function for stratified k-fold cross validation
def kfold_cross_validation(X, y):

    # Create an instance of StratifiedKFold with 5 splits
    sk_folds = StratifiedKFold(n_splits = 5)

    # Define the classifier and its model for cross-validation
    classifier = 'Optimized Count-Based Classifier'
    model = CategoricalCBC()

    # Display information about the current classifier
    print('***',classifier,'***')

    # Initialize a PrettyTable to display evaluation metrics
    table = PrettyTable()
    table.field_names = ["Evaluation Metric", "Value"]

    # Calculate and display accuracy scores during cross-validation
    accuracy = cross_val_score(model, X, y, cv=sk_folds, scoring = 'accuracy')
    table.add_row(["Accuracy Score", accuracy])
    table.add_row(['',''])
    table.add_row(["Average Accuracy Score", accuracy.mean()])
    table.add_row(['',''])

    # Calculate and display precision scores during cross-validation
    precision = cross_val_score(model, X, y, cv=sk_folds, scoring = 'precision_micro')
    table.add_row(["Precision Score", precision])
    table.add_row(['',''])
    table.add_row(["Average Precision Score", precision.mean()])
    table.add_row(['',''])

    # Calculate and display recall scores during cross-validation
    recall = cross_val_score(model, X, y, cv=sk_folds, scoring = 'recall_micro')
    table.add_row(["Recall Score",  recall])
    table.add_row(['',''])
    table.add_row(["Average Recall Score",  recall.mean()])
    table.add_row(['',''])

    # Calculate and display F1 scores during cross-validation
    f1 = cross_val_score(model, X, y, cv=sk_folds, scoring = 'f1_micro')
    table.add_row(["F1 Score", f1])
    table.add_row(['',''])
    table.add_row(["Average F1 Score",  f1.mean()])

    # Print the evaluation metrics table
    print(table)

############################################################

# Function for repeated train-test splits and model evaluation
def repetition(X, y):

    # Create an empty list to store accuracy scores
    accuracies = []

    # Iterate through 100 random states for train-test splits
    for i in range(100):
        
        # Split the dataset with an 80:20 train-test split ratio using a different random state each time
        X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=i)

        # Create an instance of the CategoricalCBC model
        model = CategoricalCBC()

        # Fit the model on the training data and predict the target values for the test dataset
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

        # Calculate and store the accuracy score for each iteration
        accuracies.append(accuracy_score(y_test,y_pred))

    # Print the list of accuracy scores
    print(accuracies)

    # Print the standard deviation and mean of the accuracy scores
    print("Standard Deviation:",np.std(accuracies))
    print("Mean:",np.mean(accuracies))

############################################################

# Function for finding and ranking dataset errors using cleanlab and creating the cleaned dataset
def cleanlab_label_errors(classifier, X, y):
    
    num_crossval_folds = 10
    pred_probs = cross_val_predict(classifier, X, y, cv=num_crossval_folds, method="predict_proba")
    
    ranked_label_issues = find_label_issues(labels=y.astype(int), pred_probs=pred_probs, return_indices_ranked_by="self_confidence")
    print(f"Cleanlab found {len(ranked_label_issues)} potential label errors")
    
    return ranked_label_issues    

############################################################

# Creating function for removing Label Errors from Test Dataset only for respective classifiers for Multiclass Classification
def dataset_error(X, y, X_train, X_test, y_train, y_test, models = {'Count-Based Classifier': CategoricalCBC()}):

    for classifier in models.keys():

        print('*********',classifier,'*********\n')
        
        # Finding label errors in the dataset according to classifier using cleanlabs
        indexes = cleanlab_label_errors(models[classifier],X,y)
        
        # Removing the label errors from the train dataset
        X_train1 = X_train.copy()
        y_train1 = y_train.copy()
        for indx in X_train1.index:
            if indx in indexes:
                X_train1.drop(indx, axis=0, inplace=True)
                y_train1.drop(indx, inplace=True)
    
        # Removing the label errors from the test dataset
        X_test1 = X_test.copy()
        y_test1 = y_test.copy()
        for indx in X_test1.index:
            if indx in indexes:
                X_test1.drop(indx, axis=0, inplace=True)
                y_test1.drop(indx, inplace=True)
        
        print("\nResults after removing label errors from the entire dataset:")
        display(classification_evaluation(X_train1, X_test1, y_train1, y_test1)[0])

        if len(y.unique()) == 2:

            # Correcting label errors in train dataset for datasets with binary classes
            y_train1 = y_train.copy()
            for indx in y_train1.index:
                if indx in indexes:
                    y_train1[indx] = 1 - y_train1[indx]

            # Correcting label errors in test dataset for datasets with binary classes
            y_test1 = y_test.copy()
            for indx in y_test1.index:
                if indx in indexes:
                    y_test1[indx] = 1 - y_test1[indx]

            print("\nResults after correcting label errors from the entire dataset (applicable for datasets with binary classes only):")
            display(classification_evaluation(X_train, X_test, y_train1, y_test1)[0])        

############################################################

# Creating function for removing Label Errors from Test Dataset only for respective classifiers for Multiclass Classification
def all_dataset_error(X, y, X_train, X_test, y_train, y_test, models = classifiers):

    for classifier in models.keys():

        try:

            print('*********',classifier,'*********\n')
            
            # Finding label errors in the dataset according to classifier using cleanlabs
            indexes = cleanlab_label_errors(models[classifier],X,y)
            
            # Removing the label errors from the train dataset
            X_train1 = X_train.copy()
            y_train1 = y_train.copy()
            for indx in X_train1.index:
                if indx in indexes:
                    X_train1.drop(indx, axis=0, inplace=True)
                    y_train1.drop(indx, inplace=True)
        
            # Removing the label errors from the test dataset
            X_test1 = X_test.copy()
            y_test1 = y_test.copy()
            for indx in X_test1.index:
                if indx in indexes:
                    X_test1.drop(indx, axis=0, inplace=True)
                    y_test1.drop(indx, inplace=True)
            
            print("\nResults after removing label errors from the entire dataset:")
            display(classification_evaluation(X_train1, X_test1, y_train1, y_test1)[0])
    
            if len(y.unique()) == 2:
    
                # Correcting label errors in train dataset for datasets with binary classes
                y_train1 = y_train.copy()
                for indx in y_train1.index:
                    if indx in indexes:
                        y_train1[indx] = 1 - y_train1[indx]
    
                # Correcting label errors in test dataset for datasets with binary classes
                y_test1 = y_test.copy()
                for indx in y_test1.index:
                    if indx in indexes:
                        y_test1[indx] = 1 - y_test1[indx]
    
                print("\nResults after correcting label errors from the entire dataset (applicable for datasets with binary classes only):")
                display(classification_evaluation(X_train, X_test, y_train1, y_test1)[0])

        except Exception as e:
            
            # Print any exceptions that may occur during the process for debugging purposes
            print(classifier,':',e)
            continue

############################################################

# Function for repeated train-test splits and model evaluation
def all_repetition(X, y, models = classifiers):

    for classifier in models.keys():

        try:

            print('*********',classifier,'*********\n')
            
            # Create an empty list to store accuracy scores
            accuracies = []
        
            # Iterate through 100 random states for train-test splits
            for i in range(100):
                
                # Split the dataset with an 80:20 train-test split ratio using a different random state each time
                X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=i)
        
                # Create an instance of the CategoricalCBC model
                model = models[classifier]
        
                # Fit the model on the training data and predict the target values for the test dataset
                model.fit(X_train,y_train)
                y_pred = model.predict(X_test)
        
                # Calculate and store the accuracy score for each iteration
                accuracies.append(accuracy_score(y_test,y_pred))
        
            # Print the list of accuracy scores
            print(accuracies)
        
            # Print the standard deviation and mean of the accuracy scores
            print("\nStandard Deviation:",np.std(accuracies))
            print("Mean:",np.mean(accuracies))
            print()

        except Exception as e:
            
            # Print any exceptions that may occur during the process for debugging purposes
            print(classifier,':',e)
            continue        

############################################################

# Function for stratified k-fold cross validation
def all_kfold_cross_validation(X, y, models=classifiers):

    for classifier in models.keys():

        try:

            print('*********',classifier,'*********\n')
            
            # Create an instance of StratifiedKFold with 5 splits
            sk_folds = StratifiedKFold(n_splits = 5)
        
            # Define the classifier for cross-validation
            model = models[classifier]
        
            # Initialize a PrettyTable to display evaluation metrics
            table = PrettyTable()
            table.field_names = ["Evaluation Metric", "Value"]
        
            # Calculate and display accuracy scores during cross-validation
            accuracy = cross_val_score(model, X, y, cv=sk_folds, scoring = 'accuracy')
            table.add_row(["Accuracy Score", accuracy])
            table.add_row(['',''])
            table.add_row(["Average Accuracy Score", accuracy.mean()])
            table.add_row(['',''])
        
            # Calculate and display precision scores during cross-validation
            precision = cross_val_score(model, X, y, cv=sk_folds, scoring = 'precision_micro')
            table.add_row(["Precision Score", precision])
            table.add_row(['',''])
            table.add_row(["Average Precision Score", precision.mean()])
            table.add_row(['',''])
        
            # Calculate and display recall scores during cross-validation
            recall = cross_val_score(model, X, y, cv=sk_folds, scoring = 'recall_micro')
            table.add_row(["Recall Score",  recall])
            table.add_row(['',''])
            table.add_row(["Average Recall Score",  recall.mean()])
            table.add_row(['',''])
        
            # Calculate and display F1 scores during cross-validation
            f1 = cross_val_score(model, X, y, cv=sk_folds, scoring = 'f1_micro')
            table.add_row(["F1 Score", f1])
            table.add_row(['',''])
            table.add_row(["Average F1 Score",  f1.mean()])
        
            # Print the evaluation metrics table
            print(table)
            print()

        except Exception as e:
            
            # Print any exceptions that may occur during the process for debugging purposes
            print(classifier,':',e)
            continue
        
