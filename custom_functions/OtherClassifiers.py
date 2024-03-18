# Importing necessary libraries and packages
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
from collections import Counter
import time
from joblib import Parallel, delayed, parallel_backend
#from dask.distributed import Client, LocalCluster
#from joblibspark import register_spark

############################################################

# Count-Based Classfier Implementation
class CountBasedClassifier(BaseEstimator,ClassifierMixin):
    
    def __init__(self, logic = 0):
        self.logic = logic 
        # logic = 0 (default) => total count; individual votes incase of tie
        # logic = 1 => individual votes; total count incase of tie
    
    def get_params(self, deep=True):
        return {"logic":self.logic}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        X = pd.DataFrame(X)
        
        self.countdict_ = {}
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.countdict_ = {}
        
        for feature in X:
            self.countdict_[feature] = {}
            for c in self.classes_:
                self.countdict_[feature][c] = Counter(X[y==c][feature])
                
        return self
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = pd.DataFrame(X)
        
        n_features_X = X.shape[1]
        
        if n_features_X != self.n_features_in_:
            raise ValueError("Expected input with %d features, got %d instead" % (self.n_features_in_, n_features_X))
            
        pred_probs = []
        
        for index in X.index:
            class_count = {c:0 for c in self.classes_}        
            feature_vote = {c:0 for c in self.classes_}
            
            for feature in X:
                max_count = 0
                vote = 0
                
                for c in self.classes_:
                    count = self.countdict_[feature][c][X.loc[index,feature]]
                    class_count[c] += count
                    
                    if count >= max_count:
                        max_count = count
                        vote = c
                 
                feature_vote[vote] += 1
            
            probs = []
            
            if self.logic == 0:
                if sum(class_count.values()) == 0:
                    for c in sorted(class_count.keys()):
                        probs.append(1/len(self.classes_))
                elif(len([key for key in class_count.keys() if class_count[key] == max(class_count.values())]) == 1):
                    for c in sorted(class_count.keys()):
                        probs.append(class_count[c]/sum(class_count.values()))
                else:
                    for c in sorted(feature_vote.keys()):
                        probs.append(feature_vote[c]/sum(feature_vote.values()))
            else:
                if sum(feature_vote.values()) == 0:
                    for c in sorted(feature_vote.keys()):
                        probs.append(1/len(self.classes_))
                elif(len([key for key in feature_vote.keys() if feature_vote[key] == max(feature_vote.values())]) == 1):
                    for c in sorted(feature_vote.keys()):
                        probs.append(feature_vote[c]/sum(feature_vote.values()))
                else:
                    for c in sorted(class_count.keys()):
                        probs.append(class_count[c]/sum(class_count.values()))
                
            pred_probs.append(probs)
            
        return np.array(pred_probs)      
         
    def predict(self, X):
        class_preds = self.predict_proba(X)
        classes = sorted(self.classes_)
        
        predictions =[classes[np.argmax(row)] for row in class_preds]
        
        return np.array(predictions)

############################################################

# Count-Based Classfier Implementation with njobs threading for parallelization
class ParallelizedCountBasedClassifier(BaseEstimator,ClassifierMixin):
    
    def __init__(self, logic = 0):
        self.logic = logic
        # logic = 0 (default) => total count; individual votes incase of tie
        # logic = 1 => individual votes; total count incase of tie
    
        #cluster = LocalCluster()
        #client = Client(cluster)

    
    def get_params(self, deep=True):
        return {"logic":self.logic}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def create_dict(self,feature, c, X, y):
        self.countdict_[feature][c] = Counter(X[y==c][feature])
        return self
        
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        X = pd.DataFrame(X)
        
        self.countdict_ = {}
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        self.countdict_ = {}
        
        #for feature in X:
            #self.countdict_[feature] = {}
        
        Parallel(n_jobs=-1,backend='threading')(delayed(self.create_dict)(i,j,X,y) for i in X for j in self.classes_)
        with parallel_backend(backend="dask"):
            Parallel()(delayed(self.create_dict)(i,j,X,y) for i in X for j in self.classes_)

        for feature in X:
            self.countdict_[feature] = {}
            for c in self.classes_:
                self.countdict_[feature][c] = Counter(X[y==c][feature])
       
        return self
    
    def find_pred(self, index, X):
        
        class_count = {c:0 for c in self.classes_}        
        feature_vote = {c:0 for c in self.classes_}
            
        for feature in X:
            max_count = 0
            vote = 0
                
            for c in self.classes_:
                count = self.countdict_[feature][c][X.loc[index,feature]]
                class_count[c] += count
                    
                if count >= max_count:
                    max_count = count
                    vote = c
                 
            feature_vote[vote] += 1
            
        probs = []
            
        if self.logic == 0:
            if sum(class_count.values()) == 0:
                for c in sorted(class_count.keys()):
                    probs.append(1/len(self.classes_))
            elif(len([key for key in class_count.keys() if class_count[key] == max(class_count.values())]) == 1):
                for c in sorted(class_count.keys()):
                    probs.append(class_count[c]/sum(class_count.values()))
            else:
                for c in sorted(feature_vote.keys()):
                    probs.append(feature_vote[c]/sum(feature_vote.values()))
        else:
            if sum(feature_vote.values()) == 0:
                for c in sorted(feature_vote.keys()):
                    probs.append(1/len(self.classes_))
            elif(len([key for key in feature_vote.keys() if feature_vote[key] == max(feature_vote.values())]) == 1):
                for c in sorted(feature_vote.keys()):
                    probs.append(feature_vote[c]/sum(feature_vote.values()))
            else:
                for c in sorted(class_count.keys()):
                    probs.append(class_count[c]/sum(class_count.values()))

        return probs
    
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = pd.DataFrame(X)
        
        n_features_X = X.shape[1]
        
        if n_features_X != self.n_features_in_:
            raise ValueError("Expected input with %d features, got %d instead" % (self.n_features_in_, n_features_X))

        pred_probs = Parallel(n_jobs=-1, backend='threading')(delayed(self.find_pred)(index,X) for index in X.index)
        
        #with parallel_backend(backend="dask"):
            #pred_probs = Parallel()(delayed(self.find_pred)(index,X) for index in X.index)
        
        #with parallel_backend(backend="spark", n_jobs=-1):
            #pred_probs = Parallel()(delayed(self.find_pred)(index,X) for index in X.index)
            
        return np.array(pred_probs)      
         
    def predict(self, X):
        class_preds = self.predict_proba(X)
        classes = sorted(self.classes_)
        
        predictions =[classes[np.argmax(row)] for row in class_preds]
        
        return np.array(predictions)
    
############################################################

# Naive Bayes Classifier Implementation
class NaiveBayesClassifier(BaseEstimator,ClassifierMixin):
    
    def __init__(self, alpha=1):
        # alpha = 0 => no smoothing
        # alpha = 1 (default) => laplace smoothing
        # alpha = k => add-k smoothing
        self.alpha = alpha

    def get_params(self, deep=True):
        return {"alpha":self.alpha}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        X = pd.DataFrame(X)
        
        self.prior_prob_ = {}
        self.posterior_prob_ = {}
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        
        for c in self.classes_:
            self.prior_prob_[c] = np.sum(y==c)/len(y)
            
            class_indices = np.where(y == c)[0]
            class_count = len(class_indices)
            class_data = X.iloc[class_indices]
            self.posterior_prob_[c] = {}
            
            for feature in class_data:
                feature_count = Counter(class_data[feature])
                self.posterior_prob_[c][feature] = {k:(v+self.alpha)/(class_count+self.alpha*len(feature_count)) for k,v in feature_count.items()}
                self.posterior_prob_[c][feature][-1] = self.alpha/(class_count+self.alpha*len(feature_count))
                
        return self
                
    def predict_proba(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X = pd.DataFrame(X)
        
        n_features_X = X.shape[1]
        
        if n_features_X != self.n_features_in_:
            raise ValueError("Expected input with %d features, got %d instead" % (self.n_features_in_, n_features_X))
        
        pred_probs = []
        
        for index in X.index:
            class_prob = {}
            
            for c in self.classes_:
                class_prob[c] = self.prior_prob_[c]
                
                for feature in X: 
                    if X.loc[index,feature] in self.posterior_prob_[c][feature].keys():
                        class_prob[c] *= self.posterior_prob_[c][feature][X.loc[index,feature]]
                    else:
                        class_prob[c] *= self.posterior_prob_[c][feature][-1]
                        
            class_prob = {k:v/sum(class_prob.values()) for k,v in class_prob.items()}
            
            pred_probs.append([class_prob[key] for key in sorted(class_prob.keys())])
            
        return np.array(pred_probs)

    def predict(self, X):
        class_preds = self.predict_proba(X)
        classes = sorted(self.classes_)
        
        predictions =[classes[np.argmax(row)] for row in class_preds]
        
        return np.array(predictions)

############################################################
