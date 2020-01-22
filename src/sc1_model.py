import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import random

from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit

from load_data import gene_list, sc1_rna_data
from load_data import sc1_phenotype, sc1_outcome
from feature_selection import feature_selection_wrapper


def sc1_model(rna_data, phenotype, outcome):
    """
    Parameters
    ----------
    rna_data:
        pandas DataFrame, Gene expression profiles and rows are samples, 
        columns are gene.Values are log2 normalized gene expression values
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

    target = 'SURVIVAL_STATUS'
    data = pd.merge(outcome,
                    rna_data,
                    left_index=True,
                    right_index=True)
    data = pd.merge(data,
                    phenotype,
                    left_index=True,
                    right_index=True)
    
    rna_feature = feature_selection_wrapper(data[gene_list],
                                            data[target], 
                                            target, 
                                            10, 
                                            feature_return='rf')
    feature = rna_feature+list(phenotype.columns)
    
    x, y = data[feature].values.astype(float), data[target].values.astype(float)
    # Stratified ShuffleSplit cross-validator
    n = 100
    sp = StratifiedShuffleSplit(n_splits=n, test_size=0.5, random_state=0)
    tmp_auc = []
    for train_sample, test_sample in sp.split(x, y):
        
        x_train, y_train = x[train_sample], y[train_sample]
        x_test, y_test = x[test_sample], y[test_sample]
        
        lr = LogisticRegression(solver='liblinear', max_iter=1000)

        parameter_grid = {'class_weight' : ['balanced'],
                        'penalty' : ['l2'],
                        'C' : [0.0001, 0.001, 0.01, 0.1],
                        #'l1_ratio':[0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],
                        #'solver': ['saga']
                        'solver': ['newton-cg', 'sag', 'lbfgs']
                        }

        grid_search = GridSearchCV(lr, 
                                   param_grid=parameter_grid, 
                                   cv=20, 
                                   scoring='roc_auc',
                                   n_jobs=-1)
        grid_search.fit(x_train, y_train)
        print('Logistic Regression')
        print('Best train score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))

        best_param = grid_search.best_params_
        clf = LogisticRegression(penalty=best_param['penalty'],
                                solver=best_param['solver'],
                                C=best_param['C'],
                                #l1_ratio = best_param['l1_ratio'],
                                class_weight=best_param['class_weight']).fit(x, y)
        
        print('test AUC',roc_auc_score(y_test,clf.predict_proba(x_test)[:,1]))
        y_pred = clf.predict(x_test)
        tmp_auc.append((grid_search.best_score_,roc_auc_score(y_test,clf.predict_proba(x_test)[:,1])))
    
    print('average AUC',np.mean(tmp_auc))
    model = LogisticRegression(solver='liblinear', max_iter=1000)
    grid_search = GridSearchCV(model, 
                               param_grid=parameter_grid, 
                               cv=20, 
                               scoring='roc_auc',
                               n_jobs=-1)
    grid_search.fit(x, y)
    best_param = grid_search.best_params_
    model = LogisticRegression(penalty=best_param['penalty'],
                               solver=best_param['solver'],
                               C=best_param['C'],
                               #l1_ratio = best_param['l1_ratio'],
                               class_weight=best_param['class_weight']).fit(x, y)

    return y_test, y_pred, tmp_auc[-1][-1], model, feature

if __name__ == '__main__':
    y_test, y_pred, auc, model, feature = sc1_model(sc1_rna_data, sc1_phenotype, sc1_outcome)
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