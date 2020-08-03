from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics


def select_best_classifier(n_features, features, labels):

    clf_k_list = []
    select_k = SelectKBest(f_classif, k=n_features)
    features = select_k.fit_transform(features, labels)
    scores = select_k.scores_
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)


    ## Naive Bayes
    clf_gb = GaussianNB()
    clf_gb.fit(features_train, labels_train)
    pred_gb = clf_gb.predict(features_test)
    clf_gb_dict = dict(f1_score=metrics.f1_score(pred_gb, labels_test), accuracy_score=metrics.accuracy_score(pred_gb, labels_test), n_features=n_features, features_scores=scores, clf=clf_gb)
    clf_k_list.append(clf_gb_dict)

    ## DecisionTree
    dt = DecisionTreeClassifier()
    parameters = {'criterion': ['gini', 'entropy'],'min_samples_split': [2, 3, 4, 5, 6, 7],'max_features': ['auto', 'sqrt', 'log2', None]}
    clf_dt = GridSearchCV(dt, parameters, iid='deprecated', cv=5, scoring='f1')
    # clf_dt = GridSearchCV(dt, parameters, scoring='f1')
    clf_dt.fit(features_train, labels_train)
    clf_dt = clf_dt.best_estimator_
    pred_dt = clf_dt.predict(features_test)
    clf_dt_dict = dict(f1_score=metrics.f1_score(pred_dt, labels_test), accuracy_score=metrics.accuracy_score(pred_dt, labels_test), n_features=n_features, features_scores=scores, clf=clf_dt)
    clf_k_list.append(clf_dt_dict) # list of dicts

    ## RandomForest
    rf = RandomForestClassifier()
    parameters = {'n_estimators': [2, 3, 4, 5, 6, 7],'criterion': ['gini', 'entropy'],'min_samples_split': [2, 3, 4, 5, 6, 7],'max_features': ['auto', 'sqrt', 'log2', None]}
    clf_rf = GridSearchCV(rf, parameters, iid='deprecated', cv=5, scoring='f1')
    clf_rf.fit(features_train, labels_train)
    clf_rf = clf_rf.best_estimator_
    pred_rf = clf_rf.predict(features_test)
    clf_rf_dict = dict(f1_score=metrics.f1_score(pred_rf, labels_test), accuracy_score=metrics.accuracy_score(pred_rf, labels_test), n_features=n_features, features_scores=scores, clf=clf_rf)
    clf_k_list.append(clf_rf_dict) # list of dicts

    ## KNeighbors
    knn = KNeighborsClassifier()
    parameters = {'n_neighbors': [1, 3, 5, 7, 9], 'weights':['uniform','distance']}
    clf_knn = GridSearchCV(knn, parameters, iid='deprecated', cv=5, scoring='f1')
    clf_knn.fit(features_train, labels_train)
    clf_knn = clf_knn.best_estimator_
    pred_knn = clf_knn.predict(features_test)
    clf_knn_dict = dict(f1_score=metrics.f1_score(pred_knn, labels_test), accuracy_score=metrics.accuracy_score(pred_knn, labels_test), n_features=n_features, features_scores=scores, clf=clf_knn)
    clf_k_list.append(clf_knn_dict) # list of dicts

    ## SVM
    svm = SVC()
    parameters = {'C': [1, 10, 100, 1000, 10000, 100000],'kernel': ['rbf'],'gamma':['scale', 'auto']}
    clf_svm = GridSearchCV(svm, parameters, iid='deprecated', cv=5, scoring='f1')
    clf_svm.fit(features_train, labels_train)
    clf_svm = clf_svm.best_estimator_
    pred_svm = clf_svm.predict(features_test)
    clf_svm_dict = dict(f1_score=metrics.f1_score(pred_svm, labels_test), accuracy_score=metrics.accuracy_score(pred_svm, labels_test), n_features=n_features, features_scores=scores, clf=clf_svm)
    clf_k_list.append(clf_svm_dict) # list of dicts

    # print ('\n\nclf_k_list')
    # print (clf_k_list)
    ordered_clf_k_list = sorted(clf_k_list, key=lambda x: x['f1_score'],reverse=True)
    clf_k_best = ordered_clf_k_list[0]
    # print ('\n\nordered_clf_k_list')
    # print (ordered_clf_k_list)
    # print ('\n\nclf_k_best')
    # print (clf_k_best)

    return clf_k_best
