import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier


# dict of standard models to evaluate
def define_models(models=dict()):
    # nonlinear models
    models['knn'] = KNeighborsClassifier(n_neighbors=7)
    models['cart'] = DecisionTreeClassifier()
    models['svm'] = SVC()
    models['bayes'] = GaussianNB()
    # ensemble models
    models['bag'] = BaggingClassifier(n_estimators=100)
    models['rf'] = RandomForestClassifier(n_estimators=100)
    models['et'] = ExtraTreesClassifier(n_estimators=100)
    models['gbm'] = GradientBoostingClassifier(n_estimators=100)
    print('Defined %d models' % len(models))
    return models


# evaluate a single model
def evaluate_model(trainX, trainy, testX, testy, model):
    model.fit(trainX, trainy)
    yhat = model.predict(testX)
    accuracy = accuracy_score(testy, yhat)
    return accuracy * 100.0


# evaluate a dict of models {name:object}, returns {name:score}
def evaluate_models(trainX, trainy, testX, testy, models):
    results = dict()
    for name, model in models.items():
        results[name] = evaluate_model(trainX, trainy, testX, testy, model)
        print('>%s: %.3f' % (name, results[name]))
    return results


# print and plot the results
def summarize_results(results):
    mean_scores = [(k, v) for k, v in results.items()]
    mean_scores = sorted(mean_scores, key=lambda x: x[1])
    mean_scores = list(reversed(mean_scores))
    print()
    for name, score in mean_scores:
        print('Name=%s, Score=%.3f' % (name, score))

if __name__ == "__main__":
    # load dataset
    trainX = np.genfromtxt("../data/preprocessed_for_learning/X_train.csv", dtype=float, delimiter=',', skip_header=1)
    trainy = np.genfromtxt("../data/preprocessed_for_learning/y_train.csv", dtype=float, delimiter=',', skip_header=1)
    testX = np.genfromtxt("../data/preprocessed_for_learning/X_test.csv", dtype=float, delimiter=',', skip_header=1)
    testy = np.genfromtxt("../data/preprocessed_for_learning/y_test.csv", dtype=float, delimiter=',', skip_header=1)
    # get model list
    models = define_models()
    # evaluate models
    results = evaluate_models(trainX, trainy, testX, testy, models)
    # summarize results
    summarize_results(results)