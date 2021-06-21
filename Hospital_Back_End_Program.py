#!/usr/bin/env python
# coding: utf-8

# In[40]:


import FeatureSelectors as fs
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

risk_factors_cg = pd.read_csv (r'risk_factors_cg.csv', index_col=0)
consolidated_data = pd.read_csv(r'consolidated_data.csv', index_col=0)
ls_features = fs.ls_features


# In[38]:


def p_out(consolidated_data, risk_factors_cg, given,treatment,tolerance):
    status = consolidated_data[consolidated_data[treatment].notnull()]
    data = risk_factors_cg[consolidated_data[treatment].notnull()]
    group = data.groupby(given)
    dic = group.indices
    probs = {key: [] for key in dic.keys()}
    for i in range(4):
        for key in dic.keys():
            modstat = status.iloc[dic[key]]
            filtered = modstat[modstat["Final Status (Encoded)"]==i]
            if (len(modstat.index)>tolerance):
                probs[key].append("{:.2f} {}".format(len(filtered.index)/len(modstat.index),len(modstat.index)))
            else:
                probs[key].append("{:.2f} {} {}".format(len(filtered.index)/len(modstat.index), len(modstat.index), "S"))
    return(probs)

df=[]
medlist =  ["Ribavirin","Kaletra (lopinavir/ritonavir)","Tocilizumab","Dexamethasone","Hydrocortisone","Prednisolone"]
for meds in medlist:
    probs = p_out(consolidated_data, risk_factors_cg, ['Age (Encoded)', 'hsTnl', 'CRP', 'Lymphocyte', 'Creatinine']                  , meds, 10)
    df.append(pd.DataFrame.from_dict(probs).T)

