

import os
import pandas as pd
import sklearn.metrics
import sklearn.preprocessing as preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from custom_transformer import ColumnExtractor, StandardScalerCustom, FeatureUnion
from custom_transformer import Log1pTransformer, ZeroFillTransformer, CustomImputer,WorkImputer,CountryConverter,MaritalStatusConverter,labelEnc
from custom_transformer import tfLinearClassifier, tfDNNClassifier, tfCustomDNNEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import sys
import numpy as np


def get_pipeline(model):
    """
    This function should build an sklearn.pipeline.Pipeline object to train
    and evaluate a model on a pandas DataFrame. The pipeline should end with a
    custom Estimator that wraps a TensorFlow model. See the README for details.
    
    cat ~ variable containing the categorical features to be used in our model
    num ~ variable containing the numerical features to be used in our model
    pipeline ~ object to be returned in order to train and evaluate model
    
    
    pipeline steps
    
    pip > Features > Extraction of Categorical features > Replacement of missing values > encoding
                   > Ectraction of numerical features > replacement of any missing values > Log Transformation > Scaling
        
        > Union of Features> replace any missing value> Run the model
        
   FeatureUnion : merges 2 dataframes toguether. Use to combine numerical and categorical features
   ColumnExtractor: used to pull collumn of interest from the dataframe. Those collumns will be used to train and test the model
   WorkImputer: Used to convert missing values from NaN to Unemployed
   CountryConverter: Used to convert native-coutry collumn from categorical to binary [1,0] (american, not american)
   MaritalStatusConverter: Converts the marital status to single or maried taking the binary form [1, 0] (married, single)
   ZeroFillTransformer: Used to fill numerical values missing values with 0
   Log1pTransformer: Return the natural logarithm of one plus the input array
   StandardScalerCustom: Standardize features by removing the mean and scaling to unit variance
   customImputer: replace any missing values with the most frequent value (can be removed)
    """
    
    cat=['workclass', 'education','marital-status','relationship', 'occupation', 'native-country' ]
    num=['age', 'education-num','capital-gain', 'capital-loss','hours-per-week' ]
            
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('categoricals', Pipeline([
                ('extract', ColumnExtractor(cat)),
                ('WorkImpute', WorkImputer('workclass')),
                ('occupation_Impute', WorkImputer('occupation')),
                ('country_convert', CountryConverter(col="native-country")),
                ('marital_status', MaritalStatusConverter(col= "marital-status")),
                ('labelEncoding', labelEnc())
                
            ])),
            ('numerics', Pipeline([
                ('extract', ColumnExtractor(num)),
                ('zero_fill', ZeroFillTransformer()),
                ('log', Log1pTransformer()),
                ('scale', StandardScalerCustom()),
            ]))
        ])),
        
        ('impute', CustomImputer()),
        model
    ])
    return pipeline

    pass

