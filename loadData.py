from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

if __name__ == '__main__':

    dataset = pd.read_csv('boston.txt', delim_whitespace=True, header=None)
    # print(dataset.shape)
    # dataset = dataset.sample(frac=1)  # losowo wybrane 100% próbek
    # print(dataset.shape)

    dataset.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

    for i in range(0, 506, 1):
        if dataset.iat[i, 2] < 15.:
            dataset.iat[i, 2] = 0
        else:
            dataset.iat[i, 2] = 15

    print(dataset)

    for label in dataset.columns:
        dataset[label] = LabelEncoder().fit(dataset[label]).transform(dataset[label]) #przypisanie nazw kolumn do zestawu danych


    X = dataset.drop(['INDUS'], axis=1)  # MEDV jest atrybutem predykcyjnym, usuwamy ze zbioru danych, bedziemy go przewidywac
    Y = dataset['INDUS']  # Y jest zbiorem danych do oceny modelu

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.99)  # test_size - w jakim stosunu dzielone sa dane, 30% dla danych testowych, 70% dla trenowanych
    model = DecisionTreeClassifier(criterion='entropy', max_depth=1)  # uzyta entropia do znalezienia najlepszego atrybutu
    AdaBoost = AdaBoostClassifier(base_estimator=None, n_estimators=5, learning_rate=1)  # learning rate default as 1,
    boostmodel = AdaBoost.fit(X_train, Y_train)  # dopasowanie modelu trenujacego do naszego modelu

    y_pred = boostmodel.predict(X_test)  # jaka jest przewidywana wartosc dla X_test

    predictions = metrics.accuracy_score(Y_test, y_pred)  # porownywanie przewidywanej wartosci z faktyczna
    print("The accuracy is: ", predictions * 100, "%")




