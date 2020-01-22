import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import random

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from load_data import sc3_phenotype, sc3_outcome,sc3_rna_cnv


def sc3_model(rna_cnv, phenotype, outcome):
    """
    Parameters
    ----------
    rna_cnv:
        pandas DataFrame, DNA copy number,Gene expression profiles and rows are samples, 
        columns are genomic cytobands and gene name.
    phenotype:
        pandas DataFrame, One-Hot encoding to represent clinical phenotype
        [SEX, WHO_GRADING, CANCER_TYPE],rows are samples.
    outcome:
        pandas DataFrame, survival status outcome,A value of 0 means 
        the patient was alive or censoring at last follow up.A value 
        of 1 means patient died before the last scheduled follow up.
    Returns
    -------
    y_test:
        array, last time y_test.
    y_pred:
        array, last time y_pred.
    auc:
        float, last time cross validation auc
    model:
        Logistic Regression model train by all data
    feature:
        list, feature list
    """
    feature_cnv = ['7q31.1', '7p15.3', '9p21.3', '10q21.1', '7p21.1',
                   '7q31.33', '7q34', '10q25.1', '7p11.2', '7q31.2',]
    feature_rna = ['PLAT', 'FOS', 'MS4A4A', 'COL5A1', 'GAS1', 'NTRK2', 
                   'ZBTB16', 'MAPK8IP1', 'GADD45G', 'PRKX']

    # combine sc1 and sc2 feature
    feature = feature_cnv+feature_rna+list(phenotype.columns)

    sp = StratifiedShuffleSplit(n_splits=100, test_size=0.5, random_state=0)
    
    tmp = pd.merge(rna_cnv,outcome, left_index=True, right_index=True)
    tmp = pd.merge(tmp, phenotype, left_index=True, right_index=True)
    # 100 Stratified ShuffleSplit cross-validator
    x, y = tmp[feature].values, tmp['SURVIVAL_STATUS'].values
    auc = []
    for train_sample, test_sample in sp.split(x, y):    
        X_train, y_train = x[train_sample], y[train_sample].astype(int)
        X_test, y_test = x[test_sample], y[test_sample].astype(int)

        lr = LogisticRegression(solver='liblinear', max_iter=1000)
        
        parameter_grid = {'class_weight' : ['balanced'],
                          'penalty' : ['l2'],
                          'C' : [0.0001, 0.001, 0.01],
                          #'l1_ratio':[0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                          #'solver': ['saga'],
                          'solver': ['newton-cg', 'sag', 'lbfgs']}
        
        grid_search = GridSearchCV(lr, 
                                   param_grid=parameter_grid, 
                                   cv=20, 
                                   scoring='roc_auc',
                                   n_jobs=-1)
        
        grid_search.fit(X_train, y_train)
        #print('Logistic Regression')
        #print('Best train score: {}'.format(grid_search.best_score_))
        #print('Best parameters: {}'.format(grid_search.best_params_))

        best_param = grid_search.best_params_
        clf = LogisticRegression(penalty=best_param['penalty'],
                                solver=best_param['solver'],
                                C=best_param['C'],
                                #l1_ratio = best_param['l1_ratio'],
                                class_weight=best_param['class_weight']).fit(x, y)
        auc.append(roc_auc_score(y_test,clf.predict_proba(X_test)[:,1]))
        print('test AUC',roc_auc_score(y_test,clf.predict_proba(X_test)[:,1]))
        y_pred = clf.predict(X_test)

    print('average AUC',np.mean(auc))
    model = LogisticRegression(solver='liblinear', max_iter=1000)
    grid_search = GridSearchCV(model, 
                               param_grid=parameter_grid, 
                               cv=20, 
                               scoring='roc_auc',
                               n_jobs=-1)
    grid_search.fit(X_train, y_train)
    best_param = grid_search.best_params_
    print(best_param)
    model = LogisticRegression(penalty=best_param['penalty'],
                               solver=best_param['solver'],
                               C=best_param['C'],
                               #l1_ratio = best_param['l1_ratio'],
                               class_weight=best_param['class_weight']).fit(x, y)
    # return last time result and model 
    return y_test, y_pred,auc[-1], model, feature


if __name__ == '__main__':
    y_test, y_pred, auc, model, feature = sc3_model(sc3_rna_cnv, sc3_phenotype, sc3_outcome)
    
    # Model Performance
    [[tn,fp],[fn,tp]] = confusion_matrix(y_test, 
                                         y_pred)
    #print confusion matrix
    print('T\\P\tAlive\tDied\tSum\nAlive\t{}\t{}\t{}\nDied\t{}\t{}\t{}\nSum\t{}\t{}\t{}\n'.format(tn,fp,tn+fp,
                                                                                        fn,tp,fn+tp,
                                                                                        tn+fn,fp+tp,tn+fp+fn+tp))
    print(classification_report(y_test, y_pred))
    print('Auc score',auc)
    print('Overall accuracy:',accuracy_score(y_test, y_pred,normalize=True))
    print('Feature list:{}'.format(feature))