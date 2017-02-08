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
from sklearn.cluster import KMeans

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

# By visually understanding the dataset, I found out TOTAL and THE TRAVEL AGENCY IN THE PARK looks to be outliers
# that are not relevant

del data_dict["TOTAL"]
del data_dict["THE TRAVEL AGENCY IN THE PARK"]
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

# FEATURE SELECTION
# Feature Selection was done by taking using SelectKBest features,
# Then precision and recall was calculated for untuned DecisionTreeClassifier
# It was observed that top 7 features give highest precision and recall
# These 7 features were used with tuned classifiers to identify the best classifier.
### Next step is to select top 7 features using SelectKBest features
def get_plot(feature, score):
    y_pos = np.arange(len(feature))
    plt.barh(y_pos,score, align='center', alpha=0.5)
    plt.yticks(y_pos, feature)
    plt.xlabel("Score")
    plt.title("Scores of each feature")
    plt.show()

### From the function below, I found the optimal precision and recall at k=7.
### Above 7, the gain in precision and recall values is less. Hence it is optimal to use
## this value of k.

def get_k_best(labels, features, k):
    kbest = SelectKBest(k=k)
    kbest.fit(features, labels)
    score = kbest.scores_
    unsorted_features = zip(features_list[1:], score)
    sorted_features = list(reversed(sorted(unsorted_features, key=lambda x: x[1])))
    feature = []
    score = []
    print sorted_features
    k_best_features = dict(sorted_features[:k])
    top_k = sorted_features[:k]
    for i in range(0,k):
        feature.append(top_k[i][0])
        score.append(top_k[i][1])
    get_plot(feature,score)
    
    return k_best_features

### Following are the scores of features
# [('exercised_stock_options', 25.097541528735491), ('total_stock_value', 24.467654047526398), ('bonus', 21.060001707536571), ('salary', 18.575703268041785), ('ratio_to_poi', 16.641707070468989), ('asset', 15.554588511146999), ('deferred_income', 11.595547659730601), ('long_term_incentive', 10.072454529369441), ('restricted_stock', 9.3467007910514877), ('total_payments', 8.8667215371077717), ('shared_receipt_with_poi', 8.7464855321290802), ('loan_advances', 7.2427303965360181), ('expenses', 6.2342011405067401), ('other', 4.204970858301416), ('ratio_from_poi', 3.2107619169667441), ('deferral_payments', 0.2170589303395084), ('restricted_stock_deferred', 0.06498431172371151)]

best_features = get_k_best(labels, features, 7)
### By running the above function and from the plot, we can say with guarantee that top 6 features would give optimal algorithm.
my_feature_list = best_features.keys()
my_feature_list.append('poi')

### Extract the features specified in features_list
data = featureFormat(my_dataset,my_feature_list)
labels, features = targetFeatureSplit(data)

### FEATURE SCALING
# Feature scaling is important to bring all features on the same scale. This is important in this case because
# there are varied scales like salary and ratio_from_poi. This will make salary to make very high effect on 
# classifier accuracy. To avoid this, classifer is very important.
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
        recall.append(recall_score(labels_test, predictions, average='micro'))
        
        

    print "precision: {}".format(np.mean(precision))
    print "recall:    {}".format(np.mean(recall))
    best_params = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
            print '%s=%r, ' % (param_name, best_params[param_name])


### Gaussian Naive Bayes
# clf = GaussianNB()
# parameters = {}
# grid_search = GridSearchCV(clf, parameters)
# evaluate_classifier(grid_search, features, labels, num_iters=100, test_size=0.2)

## Results of Gaussian Naive Bayes
# precision: 0.323571428571
# recall:    0.323571428571


### K-Means Clustering
# clf = KMeans(n_clusters=2, tol=0.001)
# parameters = {}
# grid_search = GridSearchCV(clf, parameters)
# evaluate_classifier(grid_search, features, labels, num_iters=100, test_size=0.2)

## Results of K-Means Clustering
# precision: 0.256071428571
# recall:    0.256071428571

### Decision Tree Classifier
clf = tree.DecisionTreeClassifier()
parameters = {'criterion': ['gini'],
               'min_samples_split': [2],
               'max_depth': [None],
               'min_samples_leaf': [5],
               'max_leaf_nodes': [None]}
grid_search = GridSearchCV(clf, parameters)
evaluate_classifier(grid_search, features, labels, num_iters=100, test_size=0.2)

### VALIDATION AND EVALUATION
# Precision and Recall were used as evaluation parameters.
# Precision is the fraction of retrieved instances that are relevant
# Recall is the fraction of relevant instances that are retrieved.

#  


### Results of Decision Tree Classifier
# precision: 0.328928571429
# recall:    0.328928571429
# criterion='gini',
# max_depth=None,
# max_leaf_nodes=None,
# min_samples_leaf=5,
# min_samples_split=2,

# Selected Classifier : Decision Tree Classifier
# Dump your Classifier
pickle.dump(clf, open("my_classifier.pkl", "w"))
pickle.dump(my_dataset, open("my_dataset.pkl", "w"))
pickle.dump(my_feature_list, open("my_feature_list.pkl", "w"))