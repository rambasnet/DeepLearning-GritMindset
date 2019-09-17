import csv
import sys
import numpy
import pandas as pd
import operator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from sklearn.model_selection import StratifiedKFold
from keras import backend as K

back = K.backend()
print('keras backend is', back)
# assert K.backend() == 'tensorflow', 'set backend to theano in ~/.keras/keras.json file'

data_x = None
data_y = None


def getData():
    global data_x
    global data_y
    inputFile = './data/GritMindset.csv'
    data = pd.read_csv(inputFile)
    # print([data[:]])
    label = 'HonorsScience'
    lblTypes = set(data[label])
    lblTypes = dict(zip(lblTypes, [0] * 2))
    lblTypes[2] = 1
    data[label] = data[label].map(lblTypes)
    data_y = data.pop('HonorsScience').values
    data_x = data.values

    # print(data_x[:2])


# k-fold cross validation:
# https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/


def experiment(optimizer, epochs=12, batch_size=1):
    seed = 7
    numpy.random.seed(seed)
    # defin 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(data_x, data_y):
        # create model
        model = Sequential()
        model.add(Dense(32, activation='relu', input_dim=len(data_x[0])))
        model.add(Dense(1, activation='sigmoid'))
        # compile model
        model.compile(optimizer=optimizer,
                      loss='binary_crossentropy', metrics=['accuracy'])
        # train the model, iterating on the data in batches of batch_size

        model.fit(data_x[train], data_y[train],
                  epochs=epochs, batch_size=batch_size, verbose=0)
        # evaluate the model
        scores = model.evaluate(data_x[test], data_y[test], verbose=0)
        #print('{}: {:.2f}%'.format(model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1]*100)
    return numpy.mean(cvscores), numpy.std(cvscores)


def main():
    optimizers = ['adadelta', 'adagrad', 'adam', 'rmsprop', 'sgd']
    batch_sizes = [2 ** n for n in range(0, 4)]
    outFile = 'keras-{}-results.txt'.format(back)
    getData()
    with open(outFile, 'a') as fout:
        # fout.write('optimizer: adam\nTop 5 results of optimizations\n')
        for optimizer in optimizers:
            #results = []
            # for epoch in range(9, 13):
            # for batch in batch_sizes:
            # results.append(experiment('adam', epoch, batch))
            acc, std = experiment(optimizer)
            #results.append((acc, std))
            # results.sort(reverse=True)
            fout.write('optimizer: {}: acc: {:.2f} SD: {:.2f}\n'.format(
                optimizer, acc, std))
            fout.flush()
            # break


if __name__ == "__main__":

    # main(sys.argv[1])
    main()
