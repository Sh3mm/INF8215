"""
Team:
<<<<< TEAM NAME >>>>>
Authors:
<<<<< NOM COMPLET #1 - MATRICULE #1 >>>>>
<<<<< NOM COMPLET #2 - MATRICULE #2 >>>>>
"""

BEANS = ['SIRA','HOROZ','DERMASON','BARBUNYA','CALI','BOMBAY','SEKER']

from bean_testers import BeanTester
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class MyBeanTester(BeanTester):
    def __init__(self):
        # TODO: initialiser votre modèle ici:
        self.base_clf = RandomForestClassifier(n_estimators=100)

    def train(self, X_train, y_train):
        """
        train the current model on train_data
        :param X_train: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :param y_train: 2D array of labels.
                each line is a different example.
                the first column is the example ID.
                the second column is the example label.
        """
        # TODO: entrainer un modèle sur X_train & y_train

        # transform to pandas df for easy manipulations
        train_data = [row + [tag[1]] for row, tag in zip(X_train, y_train)]
        df = pd.DataFrame(train_data).drop([0, 1, 3, 4, 5, 6, 7, 9, 10, 15], axis=1)
        for col in [i for i in df.columns if i != 17]:
            df[col] = pd.to_numeric(df[col])

        # training the base classifier
        self.base_clf.fit(df.drop(columns=[17]), df[17])

    def predict(self, X_data):
        """
        predict the labels of the test_data with the current model
        and return a list of predictions of this form:
        [
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            [<ID>, <prediction>],
            ...
        ]
        :param X_data: 2D array of data points.
                each line is a different example.
                each column is a different feature.
                the first column is the example ID.
        :return: a 2D list of predictions with 2 columns: ID and prediction
        """
        # TODO: make predictions on X_data and return them
        df_o = pd.DataFrame(X_data)
        df = df_o.drop([0, 1, 3, 4, 5, 6, 7, 9, 10, 15], axis=1)
        for col in [i for i in df.columns if i != 17]:
            df[col] = pd.to_numeric(df[col])

        # training the base classifier
        res = self.base_clf.predict(df)

        return list(zip(df_o[0].tolist(), res.tolist()))
