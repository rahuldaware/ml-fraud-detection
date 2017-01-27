# Fraud Detection from Enron Email Dataset

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives.  

This project uses scikit-learn library to perform many tasks. These include feature scaling, supervised classification, selecting K best features, validation and evaluation metrics.

### Relevant Files
1. poi_id.py : Code Implementation for entire project  
2. my_classifier.pkl : Pickle file for designed classifier  
3. my_dataset.pkl : Pickle file for dataset with new features  
4. my_feature_list.pkl : List of identified features from raw dataset  


### Results

The following results were obtained.

1. Gaussian Naive Bayes Classifier was used. With this classifier, we ran the classifier with 1000 random splitting of dataset. The mean of precision and recall was taken. The mean precision and recall values came up to be 0.22.  
2. Decision Tree Classifier was used to compare the results. With this classifier untuned, we go a poor result. The mean precision and recall came up to be 0.09.
3. After tuning the decision tree classifier with test_size=0.18, min_samples_split=10000, criterion='entropy',splitter='random', max_features='auto', we found that the precision and recall values went up. They came up to be 0.375.
