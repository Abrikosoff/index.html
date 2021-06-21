#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder

risk_factors_cg = pd.read_csv (r'risk_factors_cg.csv', index_col=0)
consolidated_data = pd.read_csv(r'consolidated_data.csv', index_col=0)


# In[2]:


enc = OrdinalEncoder()
enc.fit(np.asarray(consolidated_data[['Mortality 30 days (after discharged)']]).reshape((-1,1)))
da_enc = enc.transform(np.asarray(consolidated_data[['Mortality 30 days (after discharged)']]).reshape((-1,1)))
y_aux=da_enc
y1 = np.ravel([y_aux[i] for i in range(len(risk_factors_cg.index))])

#0-3 outcomes
y2=consolidated_data[['Final Status (Encoded)']]


# In[3]:


"""Feature selection with Mutual_info_classif"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot

def fs(X,y):
    def prepare_inputs(X_full):
        oe = OrdinalEncoder()
        oe.fit(X_full)
        X_enc = oe.transform(X_full)
        return X_enc

    # prepare target
    def prepare_targets(y_train, y_test):
        le = LabelEncoder()
        le.fit(y_train)
        y_train_enc = le.transform(y_train)
        y_test_enc = le.transform(y_test)
        return y_train_enc, y_test_enc

    # feature selection
    def select_features(X_train, y_train, X_test):
        fs = SelectKBest(score_func=mutual_info_classif, k='all')
        fs.fit(X_train, y_train)
        X_train_fs = fs.transform(X_train)
        X_test_fs = fs.transform(X_test)
        return X_train_fs, X_test_fs, fs

    # prepare input data
    X=prepare_inputs(X)

    ###below we run the training for feature selection itrns times (iterations), to get statistics on it

    itrns=5

    tot_score=np.zeros([16,])
    # split into train and test sets
    for i in range(itrns):
        # split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=546)
        # prepare output data
        y_train_enc, y_test_enc = prepare_targets(y_train, y_test)
        # feature selection
        X_train_fs, X_test_fs, fs = select_features(X_train, y_train, X_test)
        #print(fs.scores_)
        tot_score=tot_score+fs.scores_


    ### below we consider the normalized sum of all features 'scores' and select the first (most importnat) 
    ### n_feat (number of features).     
    norm_tot=tot_score/itrns
    sort_norm_tot=np.sort(norm_tot)[::-1]
    ls_ind=[]
    cnt=0
    n_feat=16


    for el in sort_norm_tot:
        cnt+=1
        if cnt<=n_feat:
            index=np.where(norm_tot==el)
            ls_ind.append(index[0][0])
    ls_features=[]
    for i in ls_ind:
        ls_features.append(risk_factors_cg.columns[i])
#   print('these are the',n_feat,'features selected \n', ls_features)
    return ls_features


# In[7]:


###Classifiers using SMOTE to balance the dataset

from sklearn.ensemble import GradientBoostingClassifier
from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import ensemble
from sklearn import tree
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import plot_confusion_matrix
from imblearn.over_sampling import SMOTE


# X_fin=risk_factors_cg[['hsTnl', 'Creatinine', 'LDH', 'Age (Encoded)', 'CRP', 'Urea', 'Neutrophil', 'CK', 'WBC']]
# enc = OrdinalEncoder()
# enc.fit(np.asarray(consolidated_data[['Mortality 30 days (after discharged)']]).reshape((-1,1)))
# da_enc = enc.transform(np.asarray(consolidated_data[['Mortality 30 days (after discharged)']]).reshape((-1,1)))
# y_aux=da_enc
# y_fin = np.ravel([y_aux[i] for i in range(len(X_fin.index))])
def fsmain(X,y,class_option):
    ls_features = fs(X,y)
    class_names = []
    if(class_option == 0):
        class_names = ['Alive','Dead']
    else:
        class_names = ['ACUTE', 'CONV.', 'DEATH', 'HOME']
    X_fin=risk_factors_cg[ls_features]
    y_fin=y

    sm=SMOTE(random_state=42)

    X_sm,y_sm=sm.fit_resample(X_fin,y_fin)
#   print("shape of X pre SMOTE",X_fin.shape)
#   print("shape of X post SMOTE",X_sm.shape)

    unique, counts = np.unique(y_sm, return_counts=True)

#   print('counts of dead-alive',np.asarray((unique, counts)).T)

    X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3,random_state=10)

    #decision tree classifier

    dtc=tree.DecisionTreeClassifier(max_depth=1)
    dtc.fit(X_train,y_train)
    d=dtc.score(X_test,y_test)
    print('Decision tree result',d)
    fig, axs = plt.subplots(ncols=2,figsize=(15,10))

    titles_options = [("Confusion matrix, without normalization", None, 0),
                      ("Normalized confusion matrix", 'true', 1)]
    for title, normalize, ind in titles_options:
        disp = plot_confusion_matrix(dtc,X_test,y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     ax=axs[ind],
                                     normalize=normalize)
        disp.ax_.set_title(title)

    fig.tight_layout()
    plt.show() 


    #random forest classifier

    rfc=ensemble.RandomForestClassifier(n_estimators=2)
    rfc.fit(X_train,y_train)
    r=rfc.score(X_test,y_test)
    #print(y_test)
    print('random forest classifier',r)
    fig, axs = plt.subplots(ncols=2,figsize=(15,10))

    titles_options = [("Confusion matrix, without normalization", None, 0),
                      ("Normalized confusion matrix", 'true', 1)]
    for title, normalize, ind in titles_options:
        disp2 = plot_confusion_matrix(rfc,X_test,y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     ax=axs[ind],
                                     normalize=normalize)
        disp.ax_.set_title(title)


    fig.tight_layout()
    plt.show()  

    #gradient boosting classifier


    gbc=ensemble.GradientBoostingClassifier(n_estimators=200)
    gbc.fit(X_train,y_train)
    g=gbc.score(X_test,y_test)
    print('gradient boost classifier',g)
    fig, axs = plt.subplots(ncols=2,figsize=(15,10))

    titles_options = [("Confusion matrix, without normalization", None, 0),
                      ("Normalized confusion matrix", 'true', 1)]
    for title, normalize, ind in titles_options:
        disp3 = plot_confusion_matrix(gbc,X_test,y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     ax=axs[ind],
                                     normalize=normalize)
        disp3.ax_.set_title(title)


    fig.tight_layout()
    plt.show()


    # RUS Classifier

    base_est = GradientBoostingClassifier()
    rusboost = RUSBoostClassifier( base_estimator=base_est)
    rusboost.fit(X_train, y_train)

    #y_pred=rusboost.predict(X_test)
    russ=rusboost.score(X_test,y_test)
    print('RUS gradient boosting',russ)

    fig, axs = plt.subplots(ncols=2,figsize=(15,10))

    titles_options = [("Confusion matrix, without normalization", None, 0),
                      ("Normalized confusion matrix", 'true', 1)]
    for title, normalize, ind in titles_options:
        disp4 = plot_confusion_matrix(rusboost,X_test,y_test,
                                     display_labels=class_names,
                                     cmap=plt.cm.Blues,
                                     ax=axs[ind],
                                     normalize=normalize)
        disp4.ax_.set_title(title)

    fig.tight_layout()
    plt.show()
    return(ls_features,dtc, rfc, gbc, rusboost)


# In[5]:


ls_features, dtc1, rfc1, gbc1, rus1 = fsmain(risk_factors_cg,y1,0)
ls_features, dtc2, rfc2, gbc2, rus2 = fsmain(risk_factors_cg,y2,1)
kappa = 4


# In[ ]:




