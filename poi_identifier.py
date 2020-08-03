#!/usr/bin/python
import sys
import pickle
# add the tools directory to sys.path
# sys.path.append("../tools/")


import numpy as np
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit
from operator import itemgetter
from select_best_classifier import select_best_classifier
from tester import dump_classifier_and_data, test_classifier

## Load the dictionary containing the dataset
with open("dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## explore the data by looking at the first person
# print ('\n\n')
# print (data_dict['METTS MARK'])

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

## get all the features, 'METTS MARK' is the first person in the data_dict
features_list = list(data_dict['METTS MARK'].keys())
print ('\n\n')
print (features_list)
## remove certain features which are not interested
features_list.remove('poi')
features_list.remove('email_address')
features_list.remove('total_payments')
features_list.remove('total_stock_value')
features_list.remove('other')
print ('\n\nnumber of total features')
print (len(features_list))

## Remove features with more than 50% NaN's in the data
data_df = pd.DataFrame(data_dict).T
# print ('\n\n')
# print (data_df.shape)
data_df.replace(to_replace='NaN', value=np.nan, inplace=True)
number_of_data_for_each_feature = data_df.shape[0]
remove_feature_threshold = number_of_data_for_each_feature * 0.5
print ('\n\nnumber_of_data_for_each_feature')
print (number_of_data_for_each_feature)
for feature in features_list:
    number_of_nan = data_df[feature].isnull().sum()
    if number_of_nan > remove_feature_threshold:
        features_list.remove(feature)

## add 'poi' to the feature list as the first feature
features_list = ['poi'] + features_list
print ('\n\nupdated feature list')
print (features_list)
print ('\n\nnumber of total features')
print (len(features_list))

### Task 2: Remove outliers
data_dict.pop('TOTAL')
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')
## Store to my_dataset for easy export below.
my_dataset = data_dict


### Task 3: Create new feature(s)
## two new features are created, one as the fraction of emails form poi
## and another is the fraction of emails to poi
for name in my_dataset:

    data_point = my_dataset[name]
    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]

    if from_poi_to_this_person!="NaN" and to_messages!="NaN":
        fraction_from_poi = float(from_poi_to_this_person)/float(to_messages)
    else:
        fraction_from_poi = 0
    data_point["fraction_from_poi"] = fraction_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    if from_this_person_to_poi!="NaN" and from_messages!="NaN":
        fraction_to_poi = float(from_this_person_to_poi)/float(from_messages)
    else:
        fraction_to_poi = 0
    data_point["fraction_to_poi"] = fraction_to_poi

features_list += ['fraction_from_poi','fraction_to_poi']

print ('\n\nfeature list: ')
print (features_list)
print ('\n\n')

## Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

## this select_classifier function will tune the paramters of different
## selected algorithms and return the best clf with the highest f1-score

k_number = int(len(features_list) / 2)
print ('\n\nk_number')
print (k_number)

clf_k_best_list = []
for k in range (1, k_number): # Try sets of 1 - number_of_features / 2
    clf_k_best_list.append(select_best_classifier(k, features, labels))
# sort the classifier in descending order according to the f1-score and accuracy score
# print ('\n\n')
# print (clf_k_best_list)
ordered_clf_k_best_list = sorted(clf_k_best_list, key=lambda x: (x['f1_score'], x['accuracy_score'])  ,reverse=True) # order by f1-score and accuracy
print ('\n\nordered_clf_k_best_list')
print (ordered_clf_k_best_list)

clf_best = ordered_clf_k_best_list[0]
clf = clf_best['clf']
print ('\n\nclf_best: ')
print (clf)
print ('\n\nf1-score, accuracy_score')
print (clf_best['f1_score'],clf_best['accuracy_score'])

number_of_features = clf_best['n_features']
print ('\n\nNumber of features used: ')
print (number_of_features)


features_scores = clf_best['features_scores']
# print ('\n\nfeatures_scores: ')
# print (features_scores)

features = features_list[1:]
# print (features)
features_and_scores = []
index = 0
for feature in features:
    features_and_scores.append([feature, features_scores[index]])
    index += 1
features_and_scores = sorted(features_and_scores, key=itemgetter(1), reverse=True)
print ('\n\nFeatures and scores: ')
print(features_and_scores)


new_features_list = []
for feature in features_and_scores[:number_of_features]:
    new_features_list.append(feature[0])
print('\n\nFeatures used: ')
print(new_features_list)
new_features_list = ['poi'] + new_features_list
print('\n\nnew_features_list')
print (new_features_list)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
# dump_classifier_and_data(clf, my_dataset, new_features_list)
