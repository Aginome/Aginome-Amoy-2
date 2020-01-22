from warnings import filterwarnings
filterwarnings('ignore')

import pandas as pd
import numpy as np
import random

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



# feature selected by Recursive Feature Elimination
def feature_selection_wrapper(data, label, target, n, step=1,verbose=20,feature_return='log'):
    """Feature selected by Recursive Feature Elimination
    Parameters
    ----------
    data:
        pandas DataFrame, Feature matrix.
    label:
        pandas DataFrame, Sample label
    target:
        str, label name
    n: 
        int or None (default=5) The number of features to select.
    step: 
        int or float, optional (default=1) If greater than or equal to 1, 
        then step corresponds to the (integer) number of features to remove 
        at each iteration.If within (0.0, 1.0), then step corresponds to the
        percentage (rounded down) of features to remove at each iteration.
    verbose: 
        int (default=5) Controls verbosity of output.
    feature_return: 
        optional, feature selected model [log:Logistic Regression, rf:Random Forest Classifier]
    Returns
    -------
    rfe_feature:
        list, feature list.
    """
    assert feature_return in ['log','rf']
    tmp = pd.merge(label, data, left_index=True, right_index=True)
    X, y = tmp[data.columns].values, tmp[target].values
    random.seed(10)
    if feature_return=='log':
        rfe_selector = RFE(estimator=LogisticRegression(penalty="l1",
                                                        C=1e-1,
                                                        multi_class="ovr",
                                                        solver='liblinear',
                                                        random_state=np.random.seed(13),
                                                        class_weight="balanced"),
                            n_features_to_select=n,
                            step=step,
                            verbose=verbose)
    if feature_return=='rf':
        rfe_selector = RFE(estimator=RandomForestClassifier(n_jobs=-1,
                                                            random_state=np.random.seed(13),
                                                            class_weight="balanced"),
                           n_features_to_select=n,
                           step=step,
                           verbose=verbose)
    rfe_selector.fit(X, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = data.loc[:, rfe_support].columns.tolist()
    print(str(len(rfe_feature)), 'selected features')
    return rfe_feature


def feature_selection_chi2(data, label, n):
    x = data.values
    y = label.values
    select_feature = SelectKBest(chi2, k=n).fit(x, y)
    result = pd.DataFrame(index=data.columns,columns=['feature_score'],data=select_feature.scores_)
    return result.sort_values('feature_score',ascending=False)[:n]


# feature selected by Random Forest
def feature_selection_embeded(data, label,target, n,feature_return='embeded_rf_feature',tmp_count=1e+10):
    """
    data: pandas DataFrame,load from train_data.csv
    label: pandas DataFrame, sample label
    feature_return: optional, feature selected model: 
        ['embeded_rf_feature','embeded_lr_selector']
    ================================================
    return: feature list
    """
    tmp = pd.merge(label, data, left_index=True, right_index=True)
    X, y = tmp[data.columns].values, tmp[target].values

    assert feature_return in ['embeded_rf_feature','embeded_lr_selector']
    # feature selected by Random Forest model
    if feature_return == 'embeded_rf_feature':
        embeded_selector = SelectFromModel(RandomForestClassifier(criterion='gini',
                                                                  max_features='auto',
                                                                  random_state=np.random.seed(13),
                                                                  n_jobs=-1,
                                                                  class_weight='balanced',
                                                                  n_estimators=500), threshold='0.8*mean')
    # feature selected by Logistic Regression
    if feature_return == 'embeded_lr_selector':
        embeded_selector = SelectFromModel(LogisticRegression(penalty='l1',
                                                              solver='liblinear',
                                                              C=1e-0), 
                                           threshold='0.5*mean')
    
    embeded_selector.fit(X, y)
    embeded_support = embeded_selector.get_support()
    embeded_feature = data.loc[:, embeded_support].columns.tolist()
    
    if feature_return=='embeded_rf_feature':
        print(str(len(embeded_feature)),'Random Forest Classifier selected features')
    if feature_return=='embeded_lr_selector':
        print(str(len(embeded_feature)),'Logistic Regression selected features')
    
    if (len(embeded_feature)>n) and (tmp_count>len(embeded_feature)):
        return feature_selection_embeded(data[embeded_feature], 
                                         label,
                                         target, 
                                         n,
                                         feature_return=feature_return,
                                         tmp_count=len(embeded_feature),)
    else:
        return embeded_feature