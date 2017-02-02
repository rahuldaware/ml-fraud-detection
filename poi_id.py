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
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV
from sklearn import tree

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi'] # You will need to use more features
initial_list = features_list
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

### Next step is to select top 5 features using SelectKBest features
def get_plot(feature, score):
    y_pos = np.arange(len(feature))
    plt.barh(y_pos,score, align='center', alpha=0.5)
    plt.yticks(y_pos, feature)
    plt.xlabel("Score")
    plt.title("Scores of each feature")
    plt.show()

def get_k_best(labels, features, k):
    kbest = SelectKBest(k=k)
    kbest.fit(features, labels)
    score = kbest.scores_
    unsorted_features = zip(features_list[1:], score)
    sorted_features = list(reversed(sorted(unsorted_features, key=lambda x: x[1])))
    feature = []
    score = []
    k_best_features = dict(sorted_features[:k])
    top_k = sorted_features[:k]
    for i in range(0,k):
        feature.append(top_k[i][0])
        score.append(top_k[i][1])
    #get_plot(feature,score)
    
    return k_best_features

best_features = get_k_best(labels, features, 6)
### By running the above function and from the plot, we can say with guarantee that top 6 features would give optimal algorithm.
my_feature_list = best_features.keys()
my_feature_list.append('poi')

### Extract the features specified in features_list
data = featureFormat(my_dataset,my_feature_list)
labels, features = targetFeatureSplit(data)

### Scale features using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)


### Using Classification Algorithms

def evaluate_classifier(grid_search, features, labels, num_iters=1000, test_size=0.2):
    accuracy = []
    precision = []
    recall = []
    for iteration in range(num_iters):
        features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels,test_size=test_size, random_state=iteration)
        grid_search.fit(features_train, labels_train)
        predictions = grid_search.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions, average='micro'))
        recall.append(precision_score(labels_test, predictions, average='micro'))

    print "precision: {}".format(np.mean(precision))
    print "recall:    {}".format(np.mean(recall))
### Gaussian Naive Bayes

clf = GaussianNB()
parameters = {}
grid_search = GridSearchCV(clf, parameters)
evaluate_classifier(grid_search, features, labels, num_iters=100, test_size=0.2)