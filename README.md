# Enron_POI_identifier
Enron fraud POI identifier

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

1. Python versions 2.7
2. Library and packages: numpy, pandas, matplotlib, sklearn

## Project Motivation<a name="motivation"></a>

For this project, I was interested in using Enron financial and email dataset to build a person of interest (POI) identifier. A person of interest is anyone that is indicted, settled without admitting guilt or testified in exchange for immunity. Different classifiers are explored in order to find the best classifier with optimal features to explore the underneath pattern and identify the POIs.
This is also my first ML project to get familiar with the ML process and explore different classifiers.


## File Descriptions <a name="files"></a>

1. dataset.pkl, which is the pickle file of the Enron financial email data;
2. feature_format.py, which is the file to extract features and labels from dataset
3. select_best_classifier.py, the function that will try different classifiers and find the one with best f1 score
4. tester.py, testing file to see if the code works properly
5. .gitignore, ignore files specified in this file
6. LICENSE, MIT license
7. my_classifier.pkl, saved pickle file of the best classifier
8. my_dataset.pkl, saved cleaned dataset with outliers removed
9. my_feature_list.pkl, saved pickle file of the selected best features

## Results<a name="results"></a>

I have used Naive Bayes, Decision Tree, Random Forest, K neighbour and SVC and used recall, precision, and f1 score as evaluation metrics. The best classifier is GaussianNB() with one feature used with the f1 score 0.378.

## Licensing, Authors, and Acknowledgements<a name="licensing"></a>

Acknowledge to CMU for the data.  You can find the Licensing for the data and other descriptive information available [here](https://www.cs.cmu.edu/~./enron/).  Otherwise, feel free to use the code here as you would like!

Acknowledge to a repo I studied on Github: [here](https://github.com/alejandrodgb/identifying-enron-fraud)
