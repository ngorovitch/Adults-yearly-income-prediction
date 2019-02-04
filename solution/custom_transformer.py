# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 20:10:09 2018

@author: ngoro
"""

import numpy as np
import pandas as pd
from functools import reduce

from sklearn.base import TransformerMixin
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.base import BaseEstimator, ClassifierMixin
from custom_tf_model import customDNN
import tensorflow as tf

class DFFunctionTransformer(TransformerMixin):
    # FunctionTransformer but for pandas DataFrames

    def __init__(self, *args, **kwargs):
        self.ft = FunctionTransformer(*args, **kwargs)

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        Xt = self.ft.transform(X)
        Xt = pd.DataFrame(Xt, index=X.index, columns=X.columns)
        return Xt
    
class FeatureUnion(TransformerMixin):
    '''
    Merges two dataframes toguether. Used to combine numerical and categorical features 
    Return a Panda Datframe
    
    '''
    

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def transform(self, X):
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        return Xunion
    
class DFImputer(TransformerMixin):
    # Imputer but for pandas DataFrames

    def __init__(self, strategy='mode'):
        self.strategy = strategy
        self.imp = None
        self.statistics_ = None

    def fit(self, X, y=None):
        self.imp = Imputer(strategy=self.strategy)
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        return self

    def transform(self, X):
       
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled
    
class WorkImputer(TransformerMixin):
    '''
     Used to convert missing values from NaN to Unemployed
    '''
    
    

    def __init__(self, col):
        self.col = col


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        
        Xw = X.fillna("Unemployed")
        Xfilled = pd.DataFrame(Xw, index=X.index, columns=X.columns)
        
        return Xfilled

class CountryConverter(TransformerMixin):
    '''
     Used to convert native-coutry collumn from categorical to binary [1,0] (american, not amerian)
    '''

    def __init__(self, col):
        self.col = col


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        X[str(self.col)] = [1 if e  == "United-States" else 0 for e in X[str(self.col)]]
        Xc=X
        Xfilled = pd.DataFrame(Xc, index=X.index, columns=X.columns)
 
        return Xfilled
    
class MaritalStatusConverter(TransformerMixin):
    '''
    Converts the marital status to single or maried taking the binary form [1, 0] (married, single)
    '''

    def __init__(self, col):
        self.col = col


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        X[str(self.col)] = X[str(self.col)].replace(['Never-married','Divorced','Separated','Widowed'], 'Single') 
        X[str(self.col)] = X[str(self.col)].replace(['Married-civ-spouse','Married-spouse-absent','Married-AF-spouse'], 'Married')
        X[str(self.col)] = X[str(self.col)].map({"Married":1, "Single":0})
        X[str(self.col)] = X[str(self.col)].astype(int)
        Xrep=X
        Xrep = pd.DataFrame(Xrep, index=X.index, columns=X.columns)
        return Xrep
    
class labelEnc(TransformerMixin):
    '''
        Converts every other collumns into encoding having a number assigned instaed of a string
        This is to transform every categorical variables into numerical ones.
    '''
    def __init__(self):
        self.dv = None

    def fit(self, X, y=None):
        return self

    def transform(self, X):
    
        enc = {}
        for c in X.columns:
            if X.dtypes[c] == np.object:
                enc[c] = preprocessing.LabelEncoder()
                enc[c] = enc[c].fit_transform(X[c])
                X.drop([c], axis=1)
                X[c]= enc[c]
        labelEnc = X
        labelEnc =pd.DataFrame(labelEnc, index=X.index, columns=X.columns)
        
       # print (labelEnc)
        return labelEnc
            

class StandardScalerCustom(TransformerMixin):
    '''
    Standardize features by removing the mean and scaling to unit variance
    '''

    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled


class ColumnExtractor(TransformerMixin):
    '''
    Used to pull collumn of interest from the dataframe.
    '''
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        
        
        Xcols = X[self.cols]
        
        return Xcols


class ZeroFillTransformer(TransformerMixin):
    '''
    Fill missing numerical values with 0s
    '''
    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        
        Xz = X.fillna(value=0)
        return Xz


class Log1pTransformer(TransformerMixin):
    '''
        Return the natural logarithm of one plus the input array
    '''
    def fit(self, X, y=None):
        
        return self

    def transform(self, X):
        
        Xlog = np.log1p(X)
        return Xlog





class tfDNNClassifier(BaseEstimator, ClassifierMixin):
    '''
    Wrap a Deep Neural Network classifier to be used in model
    Parameters fixed
    '''
    def fit(self, X, y):
        self.estimator = tf.estimator.DNNClassifier(hidden_units=[64,32,16], 
                                       feature_columns=self.getFeatures(), 
                                       n_classes=2, 
                                       optimizer=tf.train.ProximalAdagradOptimizer( learning_rate=0.1, l1_regularization_strength=0.001),
                                       model_dir='graphs/dnn')

        train_input_fn = self.create_train_input_fn(X,y)
        self.estimator.train(train_input_fn, steps=30000)

        return self.estimator

    def predict_proba(self, X):

        test_input_fn = self.create_test_input_fn(X)
        self.a=np.array([ i["class_ids"][0] for i in self.estimator.predict(test_input_fn)])
        return self.a
    
    def create_train_input_fn(self,X,Y): 
        return tf.estimator.inputs.pandas_input_fn(
            x=X,
            y=Y, 
            batch_size=32,
            num_epochs=None, 
            shuffle=True)

    def create_test_input_fn(self,X):
        return tf.estimator.inputs.pandas_input_fn(
            x=X,
            num_epochs=1,
            shuffle=False)
        
    def getFeatures(self):
        self.feature_columns = [
            tf.feature_column.numeric_column('age'),
            tf.feature_column.numeric_column('education-num'),
            tf.feature_column.numeric_column('capital-gain'),
            tf.feature_column.numeric_column('capital-loss'),
            tf.feature_column.numeric_column('hours-per-week'),   
            tf.feature_column.numeric_column('native-country'),
            tf.feature_column.numeric_column('workclass'),
            tf.feature_column.numeric_column('education'),
            tf.feature_column.numeric_column('marital-status'),
            tf.feature_column.numeric_column('relationship'),
            
            
          ]
        
        return self.feature_columns
    
    
class CustomImputer(BaseEstimator, TransformerMixin):
    '''
     replace any missing values with the most frequent value
    '''
    
    def __init__(self, strategy='fill',filler='NA'):
       self.strategy = strategy
       self.fill = filler

    def fit(self, X, y=None):
       if self.strategy in ['mean','median']:
           if not all(X.dtypes == np.number):
               raise ValueError('dtypes mismatch np.number dtype is \
                                 required for '+ self.strategy)
       if self.strategy == 'mean':
           self.fill = X.mean()
       elif self.strategy == 'median':
           self.fill = X.median()
       elif self.strategy == 'mode':
           self.fill = X.mode().iloc[0]
       elif self.strategy == 'fill':
           if type(self.fill) is list and type(X) is pd.DataFrame:
               self.fill = dict([(cname, v) for cname,v in zip(X.columns, self.fill)])
       return self

    def transform(self, X, y=None):
       
       return X.fillna(self.fill)
   
class tfLinearClassifier(BaseEstimator, ClassifierMixin):
    '''
    Wrap a TensorFlow Linear Classifier to be used in model
    Parameters fixed
    '''
    
    def fit(self, X, y):
        self.estimator= tf.estimator.LinearClassifier( feature_columns=self.getFeatures(),
                                                     optimizer=tf.train.FtrlOptimizer(
                                                             learning_rate=0.1,
                                                             l1_regularization_strength=0.001 ))
        train_input_fn = self.create_train_input_fn(X,y)
        self.estimator.train(train_input_fn, steps=10000)
        return self.estimator

    def predict_proba(self, X):

        test_input_fn = self.create_test_input_fn(X)
        self.a=np.array([ i["class_ids"][0] for i in self.estimator.predict(test_input_fn)])
        return self.a
    
    def create_train_input_fn(self,X,Y): 
        return tf.estimator.inputs.pandas_input_fn(
            x=X,
            y=Y, 
            batch_size=32,
            num_epochs=None,
            shuffle=True)

    def create_test_input_fn(self,X):
        return tf.estimator.inputs.pandas_input_fn(
            x=X,
            num_epochs=1, 
            shuffle=False)
        
    def getFeatures(self):
        self.feature_columns = [
            tf.feature_column.numeric_column('age'),
            tf.feature_column.numeric_column('education-num'),
            tf.feature_column.numeric_column('capital-gain'),
            tf.feature_column.numeric_column('capital-loss'),
            tf.feature_column.numeric_column('hours-per-week'),   
            tf.feature_column.numeric_column('native-country'),
            tf.feature_column.numeric_column('workclass'),
            tf.feature_column.numeric_column('education'),
            tf.feature_column.numeric_column('marital-status'),
            tf.feature_column.numeric_column('relationship'),
            
            
          ]
        
        return self.feature_columns    

class tfCustomDNNEstimator(BaseEstimator, ClassifierMixin):
    '''
    Wrap a TensorFlow custom Classifier to be used in model
    '''
    
    def fit(self, X, y):
        my_feature_columns = []
        for key in X.keys():
            my_feature_columns.append(tf.feature_column.numeric_column(key=key))
        
        self.estimator= tf.estimator.Estimator(model_fn = customDNN, 
                                               params={
                                                        'feature_columns': my_feature_columns,
                                                        'hidden_units': [256,128,64],
                                                        'n_classes': 2,
                                                    })
        train_input_fn = self.create_train_input_fn(X,y)
        self.estimator.train(train_input_fn, steps=15000)
        return self.estimator

    def predict_proba(self, X):

        test_input_fn = self.create_test_input_fn(X)
        self.a=np.array([ i["class"] for i in self.estimator.predict(test_input_fn)])
        return self.a
    
    def create_train_input_fn(self,X,Y): 
        return tf.estimator.inputs.pandas_input_fn(
            x=X,
            y=Y, 
            batch_size=16,
            num_epochs=None,
            shuffle=True)

    def create_test_input_fn(self,X):
        return tf.estimator.inputs.pandas_input_fn(
            x=X,
            num_epochs=1, 
            shuffle=False)
        
    def getFeatures(self):
        self.feature_columns = [
            tf.feature_column.numeric_column('age'),
            tf.feature_column.numeric_column('education-num'),
            tf.feature_column.numeric_column('capital-gain'),
            tf.feature_column.numeric_column('capital-loss'),
            tf.feature_column.numeric_column('hours-per-week'),   
            tf.feature_column.numeric_column('native-country'),
            tf.feature_column.numeric_column('workclass'),
            tf.feature_column.numeric_column('education'),
            tf.feature_column.numeric_column('marital-status'),
            tf.feature_column.numeric_column('relationship'),
            
            
          ]
        
        return self.feature_columns 
