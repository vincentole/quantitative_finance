# -*- coding: utf-8 -*-
"""
Basic forecast using LR, LDA, QDA, LSVC, RSVM, RF
Exog. Vars: Lags 1-5 of Return and Volume

Input: Stock ticker
Example: forecast_1("SPY")

Output: Confusion matrix statistics

@author: vincentole
"""

import os
from datetime import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA 
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

def create_lagged_series(ticker, lags = 5):
    """Creates DataFrame that stores percentage returns of adjusted closing
    price and lagged returns"""
     
    # Get Data   
    ts = web.DataReader(ticker, "av-daily-adjusted", datetime(2000,1,1), datetime.today(), api_key=os.getenv("ALPHAVANTAGE_API_KEY"))
    
    # Create output data set
    df = pd.DataFrame(index = ts.index)
    
    # Create return today
    df["ret_l0"] = ts["adjusted close"].pct_change()*100
    
    # resolve issues with QDA for very snmall numbers
    df.loc[ abs(df["ret_l0"]) < 0.0001, "ret_l0" ] = 0.0001
    
    # Create lags
    for lag in range(1, lags+1):
        df["ret_l%s" % lag] = df["ret_l0"].shift(lag)
        
    for lag in range(1, lag+1):
        df["volume_l%s" % lag] = ts.volume.shift(lag)
        
    # Add direction column
    df["direction"] = np.sign(df["ret_l0"])
    
    # Drop return to not have look ahead bias and NAN 
    df.drop(["ret_l0"], axis = 1, inplace = True)
    df.dropna(inplace = True)
    
    return df


def forecast_1(df):
    """Create classification forecasts of 'direction' """
    output = {}
    
    # Split df into train and test
    y = df.pop("direction")
    X = df
    
    X_train,X_test,y_train,y_test = train_test_split(X, y ,train_size=0.65, shuffle = False)
    
    # Create models
    print("Hit Rates/Confusion Matrices:\n")     
    models = [("LR", LogisticRegression()),
              ("LDA", LDA()),
              ("QDA", QDA()),
              ("LSVC", LinearSVC()),
              ("RSVM", SVC(
                C=1000000.0, cache_size=200, class_weight=None,
                coef0=0.0, degree=3, gamma=0.0001, kernel="rbf",
                max_iter=-1, probability=False, random_state=None,
                shrinking=True, tol=0.001, verbose=False)
              ),
              ("RF", RandomForestClassifier(
                n_estimators=1000, criterion="gini",
                max_depth=None, min_samples_split=2,
                min_samples_leaf=1, max_features="auto",
                bootstrap=True, oob_score=False, n_jobs=1,
                random_state=None, verbose=0)
              )]
    
    # Iterate through models
    for m in models:
        # Train each model
        m[1].fit(X_train, y_train)
        
        # Predict with each model
        pred = m[1].predict(X_test)
        
        # Output hit-rate and confusion matrix
        print("%s Balanced accuracy     : %0.3f" % (m[0], balanced_accuracy_score(y_test, pred)))
        print("%s Recall and Specificity: %s" % (m[0], recall_score(y_test, pred, average=None)))
        print("%s Precision scores      : %s" % (m[0], precision_score(y_test, pred, average=None)))
        print("%s Confusion matrix\n %s" % (m[0], confusion_matrix(y_test, pred)))
        
        output[m[0]] = {"balanced_accuracy" : balanced_accuracy_score(y_test, pred),
                        "recall" : recall_score(y_test, pred, average=None),
                        "precision" : precision_score(y_test, pred, average=None)}
        
    return output


df = create_lagged_series("SPY")
out = forecast_1(df)