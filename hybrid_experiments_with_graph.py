# coding: utf-8

import pandas as pd
import ast
import requests
import numpy as np
import os
from sklearn.linear_model import SGDClassifier
from string import punctuation
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from xgboost import XGBClassifier
import multiprocessing
import warnings
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from itertools import compress, product

#warnings.filterwarnings('ignore')

level = 'main'

cpu_count = multiprocessing.cpu_count()

########################################################################################

# coding: utf-8

# Loading data
df = pd.read_pickle('all_with_hist_emb.p').fillna(0)
print(len(df), 'rows loaded')

# converting features
features = {}
features['time_features'] = ['weekday', 'hour', 'month']
one_hot = features['time_features']

for c in one_hot:
    keep = list(df.columns)
    keep.remove(c)
    df = pd.concat([df[keep],pd.get_dummies(df[c], prefix='time_'+c)],axis=1)

features['time_features'] = []
for f in df.columns:
    if 'time_' in f and 'dict' not in f:
        features['time_features'].append(f)



features['geo_features'] = []
for f in df.columns:
    if 'geo_' in f and 'dict' not in f and '_hist_' not in f:
        features['geo_features'].append(f)



features['geo_hist_features'] = []
for f in df.columns:
    if 'geo_' in f and 'dict' not in f and '_hist_' in f:
        features['geo_hist_features'].append(f)


features['weather_features'] = ['weather_IX',
       'weather_M', 'weather_R', 'weather_S', 'weather_O', 'weather_Y',
       'weather_DD', 'weather_FH', 'weather_FF', 'weather_FX', 'weather_T',
       'weather_TD', 'weather_SQ', 'weather_Q', 'weather_DR', 'weather_RH',
       'weather_P', 'weather_U']

#features['weather_features']  += ['weather_'+x for x in ['M', 'R', 'S', 'O','Y']] # binary
features['time_features']  = []
for f in df.columns:
    if 'time_' in f:
        features['time_features'].append(f)

features['text_features']  = []
for f in df.columns:
    if 'text_' in f:
        features['text_features'].append(f)

features['graph_features']  = []
for f in df.columns:
    if 'graph_' in f:
        features['graph_features'].append(f)


features['image_features'] = []
for f in df.columns:
    if 'image_feature_' in f and 'dict' not in f:
        features['image_features'] .append(f)


prob_df = pd.read_pickle('prob_train_features_'+level+'.p')
prob_df.columns = ['prob_'+x for x in prob_df.columns]

features['prob_image_features'] = []
for f in prob_df.columns:
    if 'prob_image_' in f and 'dict' not in f:
        features['prob_image_features'] .append(f)

features['prob_geo_features'] = []
for f in prob_df.columns:
    if 'prob_geo_' in f and 'hist' not in f:
        features['prob_geo_features'] .append(f)

features['prob_geo_hist_features'] = []
for f in prob_df.columns:
    if 'prob_geo_hist_' in f and 'dict' not in f:
        features['prob_geo_hist_features'] .append(f)

features['prob_text_features'] = []
for f in prob_df.columns:
    if 'prob_text_' in f and 'dict' not in f:
        features['prob_text_features'] .append(f)

features['prob_graph_features'] = []
for f in prob_df.columns:
    if 'prob_graph_' in f and 'dict' not in f:
        features['prob_graph_features'] .append(f)

features['prob_time_features'] = []
for f in prob_df.columns:
    if 'prob_time_' in f and 'dict' not in f:
        features['prob_time_features'] .append(f)

features['prob_weather_features'] = []
for f in prob_df.columns:
    if 'prob_weather_' in f and 'dict' not in f:
        features['prob_weather_features'] .append(f)

print(list(features.keys()),'are the loaded feature groups')

keep = ['sub','main']
for f in features.keys():
    print(len(features[f]),f)
    try:
        print(list(set(features[f]))[:8])
    except:
        pass
    print()
    if 'prob' not in f:
        keep+=features[f]
df = df[keep]

print(len(df.columns), 'features loaded')

######################

######################


# Get some classifiers to evaluate
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
seed = 1075
np.random.seed(seed)
# Create classifiers


def get_results(model):
    predict = model.predict(test_features)
    rep = classification_report(test_labels,predict, output_dict=True)

    precision = precision_score(test_labels,predict, average='weighted')
    recall = recall_score(test_labels,predict, average='weighted')
    f1 = f1_score(test_labels,predict, average='weighted')

    accuracy = accuracy_score(test_labels,predict)
    print('Test Precision',str(round(precision,3)) )
    print('Test Recall',str(round(recall,3)) )
    print('Test Accuracy',str(round(accuracy,3)) )
    print('Test F1',str(round(f1,3)) )
    print()
    return {'accuracy':accuracy,'precision':precision,'recall':recall,'f1':f1, 'rep':rep}

######################


def combinations(items):
    '''
    Create all possible combinations out of a list of options
    '''

    return ( set(compress(items,mask)) for mask in product(*[[0,1]]*len(items)) )





# all features
feats = features.keys()

feats_combos = [list(x) for x in list(combinations(feats))]


print(len(feats_combos))
new_feats_combos = []
for f in feats_combos:
    new = []
    for a in f:
        typ = a.replace('prob_','')
        org = a
        if typ not in [x.replace('prob_','') for x in new]:
            new.append(org)
            
    prob = 0
    not_prob = 0
    for a in new:
        if 'prob' in a:
            prob+=1
        else:
            not_prob+=1
    if prob != 0 and not_prob != 0:
        
        new_feats_combos.append(new)
        
b = list()
for sublist in new_feats_combos:
    sublist.sort(key=len,reverse=False)
    if sublist not in b:
        b.append(sublist)

feats_combos = b
feats_combos.sort(key=len,reverse=True)

# REMOVE ALL JUST PROB OR NOT PROB
feats_combos = [['graph_features']]

print('Combinations:')
for f in feats_combos:
    print(f)       
print(len(feats_combos),'combinations')



results_df = pd.read_csv('xgboost_combinations_results_gpu_'+level+'_hybrid.csv')

done = [sorted(ast.literal_eval(x[1])) for x in results_df.to_dict()['clf'].items()]
print('Features: ',)

for a in list(feats_combos):
    print(level,a)

    if sorted(a) in done:
        print('done')
    else:




        labels = df[level]

        #le = preprocessing.LabelEncoder()
        #le.fit(labels)
        #labels = le.transform(labels) 
        features_ = []
        for k in a:

            features_+=features[k]

        train_features, test_features, train_labels, test_labels = train_test_split(df, labels, test_size = 0.2,shuffle=True,random_state=1337)
        df_prob = pd.read_pickle('prob_train_features_'+level+'.p')
        df_prob.columns = ['prob_'+x for x in df_prob.columns]
        train_features = train_features.reset_index().join(df_prob)[features_]


        df_prob = pd.read_pickle('prob_test_features_'+level+'.p')
        df_prob.columns = ['prob_'+x for x in df_prob.columns]
        test_features = test_features.reset_index().join(df_prob)[features_]

        model =  XGBClassifier(tree_method='gpu_hist',min_child_weight=10,gamma=0.5,subsample=0.6,colsample_bytree=0.6,n_estimators=200,max_depth=3)

        model.fit(train_features, train_labels)

        predict = model.predict(test_features)

        results = get_results(model)

        results['clf'] = a

        results['level'] = level
        #results['best_parameters'] = model.best_params_
        results_df = results_df.append(results, ignore_index=True)
        print(len(results_df),level,'results saved out of',len(list(feats_combos)[:-1]))
        results_df.to_csv('xgboost_combinations_results_gpu_'+level+'_hybrid.csv',index=False)   
