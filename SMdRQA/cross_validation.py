import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

def feature_selection(x_train, y_train, features, y_name, inner_splits, select_k=False, repeats=5): # inner loop function
    '''
    Function used for feature selection in nested CV, using best subset setection and average cross validation accuracy
    Input:____________________________________________________________________________________________________________________________________________
    x_train                   : training set from the outer loops of nested cross validation function
    y_train                   : output variables for the training set from the outer loops of the nested cross validation
    features                  : total set of features from which features are being selected
    y_name                    : column name for the outcome variable
    inner_splits              : cross validation splits for feature selection
    select_k                  : whether the user wants to tune k also based on training set, default = False, k=5
    repeats                   : number of repeats for cross validation for feature selection
    Output:_____________________________________________________________________________________________________________________________________________
    best_features             : list of features got selected
    best_score                : avg accuracy for the best feature subset
    best_roc_auc              : avg ROC AUC for the best feature subset
    '''
    from tqdm import tqdm
    # Load the data into a pandas DataFrame
    
    if select_k == False: 
      X=x_train
      Y=y_train
      # Split the data into features and outcome variable=

      # Define the number of features to consider in each subset
      k_values = range(1, len(X.columns) + 1)

      # Define the cross-validation scheme
      cv = RepeatedStratifiedKFold(n_splits=inner_splits, n_repeats=repeats,random_state=1)

      # Initialize variables to store the best feature set and its score
      best_score = 0
      best_roc_auc=0
      best_features = None

      # Iterate over different feature sets and evaluate their performance
      for k in tqdm(k_values):
          for feature_set in combinations(X.columns, k):
              X_sel = X[list(feature_set)]

              # Evaluate the score using cross-validation
              scores = []
              roc_aucs=[]
              for train_idx, test_idx in cv.split(X_sel, Y):
                  X_train, X_test = X_sel.iloc[train_idx], X_sel.iloc[test_idx]
                  y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

                  knn = KNeighborsClassifier()
                  knn.fit(X_train, y_train)
                  score = accuracy_score(y_test, knn.predict(X_test))
                  roc_auc = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])
                  scores.append(score)
                  roc_aucs.append(roc_auc)

              # Calculate the mean score across cross-validation repeats
              mean_score = sum(scores) / len(scores)
              mean_roc_auc= sum(roc_aucs) / len(roc_aucs)

              # Update the best feature set if necessary
              if mean_score > best_score:
                  best_score = mean_score
                  best_features = feature_set
                  best_roc_auc= mean_roc_auc

      return list(best_features), best_score, best_roc_auc
      
    elif select_k == True:
      X=x_train
      Y=y_train
      k_max= int(len(Y)/2)
      
      
             
      
      k_values = range(1, len(X.columns) + 1)

      # Define the cross-validation scheme
      cv = RepeatedStratifiedKFold(n_splits=inner_splits, n_repeats=repeats,random_state=1)

      # Initialize variables to store the best feature set and its score
      best_score = 0
      best_roc_auc=0
      best_features = None

      # Iterate over different feature sets and evaluate their performance
      for k in tqdm(k_values):
          for feature_set in combinations(X.columns, k):
              X_sel = X[list(feature_set)]

              # Evaluate the score using cross-validation
              scores = []
              roc_aucs=[]
              for train_idx, test_idx in cv.split(X_sel, Y):
                  X_train, X_test = X_sel.iloc[train_idx], X_sel.iloc[test_idx]
                  y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]

                  knn = KNeighborsClassifier()
                  knn.fit(X_train, y_train)
                  score = accuracy_score(y_test, knn.predict(X_test))
                  roc_auc = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])
                  scores.append(score)
                  roc_aucs.append(roc_auc)

              # Calculate the mean score across cross-validation repeats
              mean_score = sum(scores) / len(scores)
              mean_roc_auc= sum(roc_aucs) / len(roc_aucs)

              # Update the best feature set if necessary
              if mean_score > best_score:
                  best_score = mean_score
                  best_features = feature_set
                  best_roc_auc= mean_roc_auc
      
      
      neighbour_values = range(1, k_max)
      
      cv1 = RepeatedStratifiedKFold(n_splits=inner_splits, n_repeats=5*repeats,random_state=1)
      
      best_score1=0
      best_roc_suc1=0
      best_neigh= None
      X_sel = X[list(best_features)]
      for neigh in tqdm(neighbour_values):
          scores1=[]
          roc_aucs1=[]
          for train_idx, test_idx in cv1.split(X, Y):
              X_train, X_test = X_sel.iloc[train_idx], X_sel.iloc[test_idx]
              y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]
              knn = KNeighborsClassifier(n_neighbors=neigh)
              knn.fit(X_train, y_train)
              score = accuracy_score(y_test, knn.predict(X_test))
              roc_auc = roc_auc_score(y_test, knn.predict_proba(X_test)[:, 1])
              scores1.append(score)
              roc_aucs1.append(roc_auc)
              
          mean_score1 = sum(scores1) / len(scores1)
          mean_roc_auc1= sum(roc_aucs1) / len(roc_aucs1)
          
          if mean_score1 > best_score1:
             best_score1 = mean_score1
             best_neigh = neigh
             best_roc_auc1= mean_roc_auc1 
             
          
          
      #best_neigh = max(best_neigh, 5) 
                
      return list(best_features), best_score, best_roc_auc, best_neigh
          
      
      

def nested_cv(data_file, features, y_name, outname, repeats=1000, inner_repeats=10, outer_splits=3, inner_splits=2):
    '''
    This is a function to run nested cross validation. The outer loop is for evaluation and inner loop for feature selection
    Input:__________________________________________________________________________________________________________________
    data_file             : pandas dataframe 
    features              : features that should be used (column names)
    y_name                : outcome column name
    outname               : Name that should be added to the output file
    repeats               : number of times the outer loops should repeat
    inner_repeat          : number of times the inner loop should repeat
    outer split           : number of cross validation splits for the outer loop
    inner_split           : number of cross validation splits for the inner loop
    Output:___________________________________________________________________________________________________________________
    CSV File              : containing cross validation accuracy and ROC AUC for each of the outer loops
    '''
    from tqdm import tqdm
    

      
    # Load the data into a pandas DataFrame
    df = data_file

    # Split the data into features and outcome variable
    X = df[features]
    y = df[y_name]

    # Define the cross-validation scheme for the outer loop
    outer_cv = RepeatedStratifiedKFold(n_splits=outer_splits, n_repeats=repeats,random_state=1)

    # Initialize variables to store the best feature set and its score
    best_score = 0
    best_features = None
    FEATURES=[]
    ACCURACY=[]
    ROC_AUC=[]
    val_ACC=[]
    val_ROC_AUC=[]
    # Iterate over the outer cross-validation splits
    for train_idx, test_idx in tqdm(outer_cv.split(X, y)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Define the cross-validation scheme for the inner loop
        best_features, best_score, best_roc_auc, best_neigh = feature_selection(X_train, y_train, features, y_name, inner_splits,select_k=True, repeats=inner_repeats)
        val_ACC.append(best_score)
        val_ROC_AUC.append(best_roc_auc)
        FEATURES.append(best_features)
        print('selected number of neighbours is:', best_neigh)
        knn = KNeighborsClassifier(n_neighbors=best_neigh)
        knn.fit(X_train[best_features], y_train)
        score = accuracy_score(y_test, knn.predict(X_test[best_features]))
        ACCURACY.append(score)
        print('accuracy:', score)
        roc_auc = roc_auc_score(y_test, knn.predict_proba(X_test[best_features])[:, 1])
        ROC_AUC.append(roc_auc)

    
    DF_out=pd.DataFrame.from_dict({'features':FEATURES, 'validation_ROC_AUC':val_ROC_AUC, 'validation_accuracy':val_ACC,'test_accuracy':ACCURACY, 'test_ROC_AUC':ROC_AUC})
    DF_out.to_csv('Nested_CV_'+outname+'_fin_results.csv')


