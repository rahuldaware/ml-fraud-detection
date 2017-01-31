#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    ###creating dataFrame from dictionary - pandas
    df = pandas.DataFrame.from_dict(data_dict, orient='index', dtype=np.float)  
    print df.describe().loc[:,['salary','bonus']]

### Task 2: Remove outliers

len_data_points = len(data_dict)
no_of_data_features = len(data_dict[data_dict.keys()[0]])

#print "Num of Data Points: ", len_data_points
#print "Num of features: ", no_of_data_features
#for feature in data_dict[data_dict.keys()[0]]:
    #print "Feature : ", feature


del data_dict["TOTAL"]
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# Our first created feature would be the total asset of each individual.
# Total asset includes salary, bonus, total stock value and exercised stock options.

for elem in my_dataset:
    person = my_dataset[elem]
    if(person['salary'] != 'NaN' and
    person['bonus'] != 'NaN' and
    person['total_stock_value'] != 'NaN' and
    person['exercised_stock_options'] != 'NaN'):

        person['asset'] = person['salary'] +  person['bonus'] + person['total_stock_value'] + person['exercised_stock_options']
    
    else:
        person['asset'] = 'NaN'

### Lets create some more features. The first is the ratio of messages to this person from Poi.
### The second is ratio of messages to Poi

for elem in my_dataset:
    person = my_dataset[elem]
    if(person['from_poi_to_this_person'] != 'NaN' and 
    person['from_this_person_to_poi'] != 'NaN' and
    person['from_messages'] != 'NaN' and
    person['to_messages'] != 'NaN'):
        
        person['ratio_to_poi'] = (float(person['from_this_person_to_poi'])/float(person['from_messages']))
        person['ratio_from_poi'] = (float(person['from_poi_to_this_person'])/float(person['to_messages']))
    else:
        person['ratio_to_poi'] = 0
        person['ratio_from_poi'] = 0

### Add newly created features to our list and some more features which we will need

features_list.append('asset')
features_list.append('ratio_to_poi')
features_list.append('ratio_from_poi')
features_list.append('shared_receipt_with_poi')
features_list.append('expenses')
features_list.append('long_term_incentive')
features_list.append('loan_advances')
features_list.append('restricted_stock')
features_list.append('restricted_stock_deferred')
features_list.append('deferred_income')
features_list.append('salary')
features_list.append('deferral_payments')
features_list.append('total_stock_value')
features_list.append('exercised_stock_options')
features_list.append('total_payments')
features_list.append('bonus')
features_list.append('other')



### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### After creating new features and adding them to list of features,
### it is time to scale features using minmaxscaler

scaler = MinMaxScaler()
features = scaler.fit_transform(features)

### Next step is to select top 5 features using SelectKBest features

kbest = SelectKBest(k=5)
kbest.fit(features, labels)

top5 = kbest.get_support()

results_list = zip(kbest.get_support(), features_list[1:], kbest.scores_)
results_list = sorted(results_list, key=lambda x: x[2], reverse=True)

### 5 Best Features chosen from above are : 
### 1. exercised_stock_options
### 2. total_stock_value
### 3. bonus
### 4. salary
### 5. ratio_to_poi

selected_features = ['exercised_stock_options','total_stock_value','bonus','salary','ratio_to_poi']

data = featureFormat(my_dataset, selected_features, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

def getPrecision(prediction, labels_test):
    return precision_score(labels_test, prediction, average='micro')

def getRecall(prediction, labels_test):
    return recall_score(labels_test, prediction, average='micro')

### Classifier 1 : Naive Bayes
precision_list = []
recall_list = []

### Before starting with classifiers, let us validate the data into training and test sets
### Also, let us run the classifier for 100 different validations and take mean value of recall and precision
for i in range(100):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state = i)
    clf = GaussianNB()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    precision_list.append(getPrecision(pred, labels_test))
    recall_list.append(getRecall(pred, labels_test))

print "For Gaussian Naive Bayesian : "
print "Precision: ", np.mean(precision_list)
print "Recall: ", np.mean(recall_list)

### Classifier 2 : Decision Tree Classifier
precision_list = []
recall_list = []
for i in range(100):
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state = i)
    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    precision_list.append(getPrecision(pred, labels_test))
    recall_list.append(getRecall(pred, labels_test))

print "For Decision Tree Classifier before tuning parameters: "
print "Precision: ", np.mean(precision_list)
print "Recall: ", np.mean(recall_list)
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.18)

#clf = GaussianNB()
clf = DecisionTreeClassifier(criterion='entropy',splitter='random',max_features='auto',min_samples_split=10000)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

print "For Decision Tree Classifier after tuning parameters: "
print "Precision: ", getPrecision(pred, labels_test)
print "Recall: ", getRecall(pred, labels_test)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)