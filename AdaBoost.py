import pandas as pd
import numpy as np

def shuffle_data(X, y, a=None): #losowanie próbek do X i Y
    np.random.seed(a) #a  is the seed value. If the a value is None, then by default, current system time is used.
    index = np.arange(X.shape[0]) #https://stackoverflow.com/questions/46611476/what-is-different-between-x-shape0-and-x-shape-in-numpy
    np.random.shuffle(index) #generate random index
    return X[index], y[index] #return arrays values of generated indexes

def train_test_split(X, y, test_size=0.3, a=None): #dzieli dane treningowe od danych testowych w stosunku podanym w test_size
    X, y = shuffle_data(X, y, a) #wylosowane próbki danych do trenowania
    split_i = len(y) - int(len(y) // (1 / test_size)) #obliczenie miejsca, gdzie dane beda dzielone
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test

dataset = pd.read_csv('boston.txt', delim_whitespace=True, header=None)
dataset = dataset.sample(frac=1) #losowo wybrane 100% próbek
dataset.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

X = dataset.drop(['CHAS'], axis=1) #CHAS jest atrybutem predykcyjnym, usuwamy ze zbioru danych, bedziemy go przewidywac
Y = dataset['CHAS'] #Y jest zbiorem danych do oceny modelu

#dane w formacie tablicy
X_array = np.array(X)
Y_array = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X_array, Y_array, test_size=0.3)
print(X_train)
