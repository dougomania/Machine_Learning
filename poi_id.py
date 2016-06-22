#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pprint
from scipy import stats
import numpy as np
import numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn import metrics 
from tester import test_classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn import tree
from collections import Counter
import copy


#For Select K Best
#from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest,f_classif

#import decision tree
from sklearn import tree

#This is used to split the data set up
from sklearn import cross_validation

#from sklearn.feature_selection import chi2

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list_total =  ['poi','salary', 'to_messages', 'deferral_payments', 'total_payments', \
                 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', \
                 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', \
                 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', \
                 'from_poi_to_this_person','other']       
        
    
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    #print data_dict.keys()
    data_dict.pop("TOTAL", None) 
    data_dict.pop('THE TRAVEL AGENCY IN THE PARK', None)
    

print data_dict.values()
### Task 1: Looking at Dataset to remove outliers

# Delete NaN values
# We will delete those from the data who have over 90% of values that are NaN
# This is 1 person 
## Deleting people who have too many NaN values 
nans =[]
nan_total =[]
value_count={}
poi_count = 0
for key in data_dict:
    count = 0
    for values in data_dict[key]:   
              
        #print data_dict[key][values]
        if data_dict[key][values] == 'NaN':
            #Counts the amount of Nans per each variable 
            if value_count.get(values) != None:
                value_count[values] = value_count[values] + 1
            else:
                value_count[values] = 1
            #counts the amount of nas per key 
            count = count + 1
            
        if values == "poi" :
            if data_dict[key][values] == True:
                poi_count += 1
         
                
        nan_total.append(count)
    if count >= 17:
        nans.append(key)
        print key

print "number of POIs", poi_count
print "number of people being analysed: " , len(data_dict.keys() ) 
matplotlib.pyplot.hist(nan_total, bins=21, normed=1)        
matplotlib.pyplot.title('Distribution of Nans that each dict key has')
matplotlib.pyplot.show()

matplotlib.pyplot.bar(range(len(value_count)), value_count.values(), align='center')
matplotlib.pyplot.xticks(range(len(value_count)), value_count.keys(),rotation=90)
matplotlib.pyplot.title('The amount of nans per value in dict MAX: 144')
matplotlib.pyplot.show()


#pop the values from the previous loop
for v in nans:  
    data_dict.pop(v, None)
 
# Graphs below showed there was something weird with Kaminski
# What was discovered online is that he had multiple email addresses
# So His email data was off
# Taking him out increased data by 20%   
data_dict.pop('KAMINSKI WINCENTY J', None)
#data_dict.pop('FREVERT MARK A', None)

def make_more_than_200000():
    for k,v in data_dict.iteritems():
        if v['loan_advances'] >= 200000:
            if v['loan_advances'] != "NaN":
                print "THIS IS WHAT", k, v['deferral_payments']

###Grapsh to show outliers. Only one thing showed. KAMINSKI WINCENTY J, but
    
# looking on graphs to see if there is anything inconsistant 
def graph_loop(features_list_total):
    for fl in features_list_total:
        print fl
        features_list= ['poi','salary', fl]
        make_graph(features_list, data_dict)


def make_graph(feature_list, data_dict, low1 = None, low2 = None, up1 = None, up2 = None):    
    
    data = featureFormat(data_dict, feature_list)
    #data = data1[data[1] >= 1000000] 
    matplotlib.pyplot.scatter(data[:,1], data[:,2], c=data[:,0])
    matplotlib.pyplot.xlabel(feature_list[1])
    matplotlib.pyplot.ylabel(feature_list[2])
    matplotlib.pyplot.show()    
       
#graph_loop(features_list_total)
      
####TASK 2 PICK THE FEATURES     
# This function in tools dats the the features and makes a list
# The first feature is POI so it is 0 or 1
      
data = featureFormat(data_dict, features_list_total)
#Target split takes the POI value out and makes a array with other features
labels, features = targetFeatureSplit(data)


def select_features(features, number, new_feature = False):
    
    #if new_feature == True:
    #    features_list_total.append("Income_indicator")
    
    #kbest_v={} #creates an empty sub dictionary 
    kbest = SelectKBest(k=number)
    kbest.fit_transform(features,labels)
    #print sorted(kbest.scores_)
    
    
    what_features = kbest.get_support() #True or false of the features selected
    list_of_features = []
    index = 1

    for which in what_features:
        if which == True:
            list_of_features.append(features_list_total[index])
            index += 1
    return kbest.scores_, list_of_features

#Kbest on all the scores
kscores, list_of_features = select_features(features, len(features[0]))

#From the output I will choose the best 5
# There is a big drop off with the added feature 
# 
kscores, list_of_features = select_features(features,5)

print kscores


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
class PartialScale(BaseEstimator):
    
    def __init__(self):
        pass
    
    #X: is the array 
    #self: is the first instance??!
    def transform(self, X):
        X = np.array(X)
        
        # column with following titles are extracted
        ff1 = ['bonus','deferral_payments', 'exercised_stock_options', 'deferred_income', 'long_term_incentive']
        flt = features_list_total
        ff1i = [flt.index(ff1[0]),flt.index(ff1[1]), flt.index(ff1[2]), flt.index(ff1[3]),flt.index(ff1[4])]        
        
        cols = X[:, [ff1i[0], ff1i[1], ff1i[2],ff1i[3],ff1i[4]]]  
        min_max_scaler = MinMaxScaler()
        
        # all columns are scaled that are listed above 
        colsScaled = min_max_scaler.fit_transform(cols)
        
        #flip a few of the columns 
        def invert_MaxMin(dicts, feature):
        
            dicts[feature]=1-dicts[feature] 
            return dicts
            
        colsScaled = invert_MaxMin(colsScaled, 1)
        colsScaled = invert_MaxMin(colsScaled,3)
        colsScaled = invert_MaxMin(colsScaled,4)
        
        
        # add whatever other transformations you need
        sumCols = np.sum(colsScaled, axis=1)
        
        X = np.insert(X, 0, sumCols, axis=1)     
    
        
        return X

    def fit(self, X, y=None):
        return self
        
    #def get_params(self, deep =True):
    #    return self
     
    #def set_params(self, **parameters):
    #    return self
    

    

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#1. First do a cross validation_test_split to get a good cross section
# of the data
#add_new_feature(data_dict, features_list_total)

#1. get the data
data = featureFormat(data_dict, features_list_total)
labels, features = targetFeatureSplit(data)

#2. sample the data 
sk_fold = StratifiedShuffleSplit(labels, 100, random_state = 42)

#3 add new feature
income_scale = PartialScale()


#Select K Best and PCA
skb = SelectKBest(f_classif,k=5)
pca = PCA(n_components=2)

###COmbine the two above feature selections
combined_features = FeatureUnion([("pca", pca), ("k_select", skb)])


#Choose the classifiers
##put all three right into pipeline

#pipeline: first add teh features`
#2. Then select 5 k best
#3 Then provide a classifier
'''
####These Lines is for the GridSearchCV PCA FeatureUnion
pipe_nb_pca = Pipeline(steps=[('income_scale', income_scale),("cf", combined_features), ("NaiveBayes", GaussianNB())])
pipe_rf_pca = Pipeline(steps=[('income_scale', income_scale),("cf", combined_features), ("RandomF", RandomForestClassifier())])
pipe_dt_pca = Pipeline(steps=[('income_scale', income_scale),("cf", combined_features), ("decisionT", tree.DecisionTreeClassifier())])
'''

#These Lines are pipelines for not PCA 
pipe_nb = Pipeline(steps=[('income_scale', income_scale),("selectK", skb), ("NaiveBayes", GaussianNB())])
pipe_rf = Pipeline(steps=[('income_scale', income_scale),("selectK", skb), ("RandomF", RandomForestClassifier())])
pipe_dt = Pipeline(steps=[('income_scale', income_scale),("selectK", skb), ("decisionT", tree.DecisionTreeClassifier())])

#Parameters for PCA FeatureUnion 


param_grid_nb_pca = {'cf__k_select__k': range(3,9), 
                     'cf__pca__n_components':[2,3,4,5] }
'''
param_grid_rf_pca = {
                'cf__k_select__k': range(3,9), 
                'cf__pca__n_components':[2,3,4,5],
                'RandomF__n_estimators': [5,10],
                'RandomF__max_features': ['sqrt', 'log2']
                }

param_grid_dt_pca = {
                'cf__k_select__k': range(3,9),
                'cf__pca__n_components':[2,3,4,5],
                'decisionT__splitter':['random'],
                'decisionT__min_samples_leaf':[1, 2],
                'decisionT__min_samples_split':[1,2,4],
                'decisionT__criterion':['gini']
                }
'''
             
#Parameters for Non PCA 
param_grid_nb = {'selectK__k': range(4,6)}
'''            
param_grid_rf = {
                'selectK__k': range(3,9), 
                'RandomF__n_estimators': [5,10],
                'RandomF__max_features': ['sqrt', 'log2']
                }                            
param_grid_dt = {
                'selectK__k': range(3,9),
                'decisionT__splitter':['random'],
                'decisionT__min_samples_leaf':[1, 2],
                'decisionT__min_samples_split':[1,2,4],
                'decisionT__criterion':['gini']
                }          
'''
#NON PCA GRID
gs = GridSearchCV(pipe_nb, param_grid = param_grid_nb, scoring='f1', cv=sk_fold  )
gs.fit(features, labels)
'''
gs_rf = GridSearchCV(pipe_rf, param_grid = param_grid_rf, scoring='f1', cv=sk_fold  )
gs_rf.fit(features,labels)

gs_dt = GridSearchCV(pipe_dt, param_grid = param_grid_dt, scoring='f1', cv=sk_fold  )
gs_dt.fit(features,labels)
'''
'''#PCA GRID 
gs_dt_pca = GridSearchCV(pipe_dt_pca, param_grid = param_grid_dt_pca, scoring='f1', cv=sk_fold  )
gs_dt_pca.fit(features,labels)

gs_nb_pca = GridSearchCV(pipe_nb_pca, param_grid = param_grid_nb_pca, scoring='f1', cv=sk_fold  )
gs_nb_pca.fit(features,labels)

gs_rf_pca = GridSearchCV(pipe_rf_pca, param_grid = param_grid_rf_pca, scoring='f1', cv=sk_fold  )
gs_rf_pca.fit(features,labels)
'''



print "...........NAIVE BAIS (NOT PCA_)............ "
print "The best parameters for the grid:"
print "Naive Bais :" , gs.best_params_

# make a copy of features_list
features_list_new = copy.deepcopy(features_list_total)

# remove poi from list
features_list_new.remove('poi')

# insert new feature(s) name(s)
features_list_new.insert(0, "New Feature")



X_new = gs.best_estimator_.named_steps['selectK']
# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in X_new.scores_ ]
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  X_new.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in 'X_new.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list_new[i], feature_scores[i], feature_scores_pvalues[i]) for i in X_new.get_support(indices=True)]

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

# Print
print ' '
print 'Selected Features, Scores, P-Values'
print features_selected_tuple



clf = gs.best_estimator_

print "Tester Classification report for GridSearch and Naives Bayes:" 
test_classifier(clf, data_dict, features_list_total)
print ""
#print "PPPPPPPPCCCCCCCCCAAAAAAAAAAAA"
#test_classifier(gs_nb_pca.best_estimator_, data_dict, features_list_total)


print "k_scores"

features_list_total_1 =  ['New Feature','salary', 'to_messages', 'deferral_payments', 'total_payments', \
                 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', \
                 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', \
                 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', \
                 'from_poi_to_this_person'] 
print len(features_list_total_1)
features_selected_bool = clf.named_steps['selectK'].get_support()
print len(features_selected_bool)
k_scores = clf.named_steps['selectK'].scores_
features_selected_list = [x for x, y,z in zip(features_list_total_1[0:], features_selected_bool, k_scores) if y]
features_selected_score = [z for x, y,z in zip(features_list_total_1[0:], features_selected_bool, k_scores) if y]
print features_selected_list
print features_selected_score

'''
####This line is for just a regular classifier no GRID
#clf = Pipeline(steps=[('income_scale', income_scale),("SKB", skb), ("NaiveBayes", GaussianNB())])
#test_classifier(clf, data_dict, features_list_total)

print "-----------RANDOM FOREST (NOT PCA)-------------"
print "Random Forest:", gs_rf.best_params_
print "Results........................."
clf1 = gs_rf.best_estimator_
test_classifier(clf1, data_dict, features_list_total)
print ""
#print "PPPPPPPCCCCCCCCAAAAAA"
#test_classifier(gs_rf_pca.best_estimator_, data_dict, features_list_total)


print "=========Decision Tree Classifier PCA===========" 
clf2 = gs_dt.best_estimator_
test_classifier(clf2, data_dict, features_list_total)

#print "PPPPPPPPCCCCCCCCCAAAAAAAAAAAA"
#test_classifier(gs_dt_pca.best_estimator_, data_dict, features_list_total)
'''


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, data_dict, features_list_total)
