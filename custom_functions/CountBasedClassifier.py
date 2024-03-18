"""
The module :mod:`classifiers` implements the count-based classification 
logic (currently only for categorical data).
"""

# Authors: Sanskriti Sanjay Kumar Singh <singhsanskriti2112@gmail.com>
#          Alok Chauhan <alok.chauhan@vit.ac.in>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from collections import Counter

class CategoricalCBC(BaseEstimator,ClassifierMixin):
    """ 
    Count-Based Classifier for Categorical Data

    For more information regarding how the classifier works, read more
    in the :ref:`User Guide <user_guide>`.

    Parameters
    ----------
    verbose : bool, default='False'
        Controls the verbosity when fitting (i.e. displays the Count 
        Lookup Dictionary when `verbose` is True).

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        The classes seen during :term:`fit`.

    countdict_: dict
        A nested dictionary structure that stores the counts of each 
        distinct category within each feature and class label. 

    n_features_in: int
        Number of features seen during :term:`fit`.

    Examples
    --------
    >>> from countlib.classifiers import CategoricalCBC
    >>> import numpy as np
    >>> X = np.random.choice(range(0,10),(100,6))
    >>> y = np.random.choice([0, 1], size=100)
    >>> estimator = CategoricalCBC()
    >>> estimator.fit(X, y)
    >>> print(estimator.predict([3, 6, 7, 2, 3, 4]))
    """
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def get_params(self, deep=False):
        """ Get parameters of this classifier."""
        return {"verbose":self.verbose}
    
    def set_params(self, **parameters):
        """ Set the parameters of this classifier."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, X, y):
        """ 
        Create a count lookup table from the training set (X,y).
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
            
        y : array-like, shape (n_samples,)
            The target values (class labels).

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # Check and validate input data and labels
        X, y = check_X_y(X, y)
        check_classification_targets(y)

        # Convert input data to a pandas DataFrame
        X = pd.DataFrame(X)

        # Initialize attributes
        self.countdict_ = {}
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.countdict_ = {}

        # Loop through each feature in the input data
        for feature in X:
            
            # Initialize a dictionary to store counts for each category 
            # within the feature
            self.countdict_[feature] = {}

            # Loop through each class label
            for c in self.classes_:

                # Use Counter to count occurrences of each category within
                # the specific class
                self.countdict_[feature][c] = Counter(X[y==c][feature])
                
        # Print the count dictionary if verbose mode is enabled
        if self.verbose == True:
            print(self.countdict_)

        # Return the fitted instance of the classifier
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for input samples (X).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray, shape (n_samples, n_classes)
            Returns the probability of the samples for each class in the 
            model. The columns correspond to the classes in sorted order, 
            as they appear in the attribute :term:`classes_`.     
        """

        # Ensure the classifier has been fitted
        check_is_fitted(self)

        # Check and validate input data
        X = check_array(X)

        # Convert input array to a pandas DataFrame
        X = pd.DataFrame(X)

        # Check and validate the number of features in the input
        n_features_X = X.shape[1]
        if n_features_X != self.n_features_in_:
            raise ValueError("Expected input with %d features, got %d instead" 
                                % (self.n_features_in_, n_features_X))
            
        # Initialize an empty list to store predicted probabilities for 
        # each sample    
        pred_probs = []

        # Loop through each index in the input DataFrame
        for index in X.index:

            # Initialize dictionaries to store counts and votes for 
            # each class
            class_count = {c:0 for c in self.classes_}        
            feature_vote = {c:0 for c in self.classes_}

            # Loop through each feature in the input sample
            for feature in X:
                max_count = 0
                vote = 0

                # Loop through each class label
                for c in self.classes_:

                    # Count occurrences of each category within the 
                    # specific class and increment the class count
                    count = self.countdict_[feature][c][X.loc[index,feature]]
                    class_count[c] += count

                    # Update the most frequent class (vote) for the 
                    # current feature
                    if count >= max_count:
                        max_count = count
                        vote = c
                        
                # Increment the vote count for the most frequent class in
                # the feature 
                feature_vote[vote] += 1

            # Initialize an empty list to store predicted probabilities 
            # for the current sample
            probs = []

            # Calculate class probabilities based on counts and votes

            # Case: No occurrences for any class in any feature
            if sum(class_count.values()) == 0:
                # Assign equal probabilities to each class
                for c in sorted(class_count.keys()):
                    probs.append(1/len(self.classes_))

            # Case: Clear majority for one class based on counts
            elif(len([key for key in class_count.keys() if class_count[key]
                      == max(class_count.values())]) == 1):
                # Calculate probabilities based on counts of occurrences
                for c in sorted(class_count.keys()):
                    probs.append(class_count[c]/sum(class_count.values()))
            
            # Case: Tied counts for two or more classes
            else:
                # Calculate probabilities based on the votes (most frequent 
                # class for each feature)
                for c in sorted(feature_vote.keys()):
                    probs.append(feature_vote[c]/sum(feature_vote.values()))

            # Append the calculated probabilities for the current input sample
            pred_probs.append(probs)

        # Convert the list of probabilities to a numpy array and return
        return np.array(pred_probs)      
         
    def predict(self, X):
        """
        Perform classification on an array of test vectors (X).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray, shape (n_samples,)
            Predicted target values (class labels) for X.
        """

        # Get predicted probabilities using the predict_proba method
        class_preds = self.predict_proba(X)

        # Get sorted class labels
        classes = sorted(self.classes_)

        # Loop through each input sample and select the class label with 
        # the highest predicted probability as the predicted class label
        predictions =[classes[np.argmax(row)] for row in class_preds]

        # Convert the list of predicted class labels to a numpy array and return
        return np.array(predictions)

############################################################











