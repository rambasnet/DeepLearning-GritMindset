import csv
import numpy
import pandas
import operator
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle


def getData():
    inputFile = 'data/GritMindset.csv'
    data = pandas.read_csv(inputFile)
    # print([data[:]])
    data_y = data.pop('HonorsScience').values
    data_x = data.values
    return data_x, data_y


def runExperimentsAvgCV():
    model = SVC(gamma='auto')
    X, y = getData()
    kfold = model_selection.StratifiedKFold(
        n_splits=10, shuffle=True, random_state=7)
    cv_results = model_selection.cross_val_score(
        model, X, y, cv=kfold, scoring='accuracy')

    print('average accuracy = {:.2f} std. dev = {:.2f}'.format(
        cv_results.mean(), cv_results.std()))
    with open('GritMindSet-svmResults.txt', 'w') as f:
        print('average accuracy = {:.2f} std. dev = {:.2f}'.format(
            cv_results.mean(), cv_results.std()), file=f, flush=True)

    #model.fit(X, y)
    # model.save()
    #print('model = ', model)
    """
    initial_type = [('float_input', FloatTensorType([1, 4]))]
    onx = convert_sklearn(model, initial_types=initial_type)
    with open("svm_GritMindset.onnx", 'wb') as f:
        f.write(onx.SerializeToString())
    """
    with open("./models/svm_GritMindset.model", 'wb') as f:
        pickle.dump(model, f)

    # http://onnx.ai/sklearn-onnx/ - didn't work!!
    # https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/


def experimentGridSearch():
    parameters = {'kernel': ('linear', 'rbf'), 'C': [
        0.001, 0.01, 0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1, 1]}
    model = svm.SVC(gamma="scale")
    gs = model_selection.GridSearchCV(model, parameters, cv=10)
    X, y = getData()
    gs.fit(X, y)

    # grid search best results
    print(gs.best_score_, gs.best_params_)
    with open('GritMindSet-svmGridSearchResults.txt', 'w') as f:
        print(gs.best_score_, gs.best_params_, file=f, flush=True)


if __name__ == "__main__":
    # runExperimentsAvgCV()
    experimentGridSearch()
