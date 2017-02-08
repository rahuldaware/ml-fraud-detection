# Fraud Detection from Enron Email Dataset

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.  

This project uses scikit-learn library to perform many tasks. These include feature scaling, supervised classification, selecting K best features, validation and evaluation metrics.

### Relevant Files
1. poi_id.py : Code Implementation for entire project  
2. my_classifier.pkl : Pickle file for designed classifier  
3. my_dataset.pkl : Pickle file for dataset with new features  
4. my_feature_list.pkl : List of identified features from raw dataset  

### Understanding the Dataset
1. Total number of data points : 146
2. 18 points are marked as POI and 128 are marked as non POI
3. The list of features used are :  
```
['poi', 'asset', 'ratio_to_poi', 'ratio_from_poi', 'shared_receipt_with_poi',
'expenses', 'long_term_incentive', 'loan_advances', 'restricted_stock', 
'restricted_stock_deferred', 'deferred_income', 'salary',
'deferral_payments', 'total_stock_value', 'exercised_stock_options',
'total_payments', 'bonus', 'other']
```
### Feature Scores
The following are scores of features which got selected. The data was used for these features only.

| Feature     | Score |
| ---------- |:----------:|
| exercised_stock_options   | 25.097541528735491 |
| total_stock_value   | 24.467654047526398 |
| bonus   | 21.060001707536571 |
| salary  | 18.575703268041785 |
| ratio_to_poi  | 16.641707070468989 |
| asset   | 15.554588511146999 |
| deferred_income   | 11.595547659730601 |
| long_term_incentive   | 10.072454529369441 |

### Validation
After carefully observing the evaluation metrics for different test_sizes, I found 20% of the dataset as test set and 80% as training set gives optimal results. Precision and recall values were calculated for different proportion of test set and training set. I found that at 80-20 ratio, highest precision and recall was found. This parameter is very useful is tuning the evaluation of our algorithm. This will perfectly trade-off bias and variance of our algorithm.

### Feature Selection

I implemented three new features. First one is asset. This feature is the sum of salary, bonus and stock values. Second one being ratio_to_poi. This is the fraction of emails from this person to POI and total messages from this person. The last one being ratio_from_poi. This is the fraction of email to this person from POI to the total messages to this person. The precision and recall values before and after adding these features were calculated for a decision tree classifer. The values are as follows :

|  Evaluation Metrics       | Precision | Recall  |
| ------------- |:-------------:| -----:|
| Before adding new features      | 0.083 | 0.083 |
| After adding new features      |  0.2083 |   0.2083 |

The reason to add these new features is pretty obvious. They reflect to be the most important features which would signify fraudulent activities. 

### Algorithms
1. The first algorithm used was Gaussian Naive Bayes. After trying various values, I obtained best precision and recall when test_size was 0.2 for 100 iterations. The precision and recall values obtained are: precision = 0.323571428571, recall = 0.323571428571.  
2. The second algorithm tried was K-Means Clustering. For k=2 and tol=0.001, I obtained the best precision and recall values. They were precision = 0.256071428571, recall = 0.256071428571.  
3. The third algorithm was Decision Tree Classifier. The best precision and recall values were obtained for this algorithm. I tried to tune this algorithm many times. The best tuning parameters were criterion='gini', max_depth=None, max_leaf_nodes=None, min_samples_leaf=5 and min_samples_split=2. The precision and recall values obtained here were precision = 0.328928571429, recall = 0.328928571429. 
### Final Results
From the precision and recall values, the winner was Decision Tree Classifier with the tuning parameters mentioned before. We have successfully obtained a classifier which performs better than our expectation of precision and recall value above 0.3. 

All intermediate results can be obtained as comments in poi_id.py file or after running the python code.  
