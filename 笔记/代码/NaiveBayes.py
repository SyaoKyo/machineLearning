# project    :don't to try
# fileName   :NaiveBayes.py
# user       :cheng
# createDate :2020-11-12 16:12

import numpy as np
import collections
import copy
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self, X, y):

        self.X = X
        self.y = y
        self.characteristicMaxValue = []
        self.characteristicMinValue = []
        self.characteristicQuantity = len(X[0])
        self.Pxy = {}
        self.Py = {}
        self.pred_result = {}
        self.decimal_digits = 10 ** 0
        self.classification = collections.Counter(self.y)
        self.getDecimalDigits(X[0][0])
        self.characteristicValueRange()

    def getDecimalDigits(self, val: float):
        val_str = str(val)
        digits_location = val_str.find('.')
        if digits_location:
            n = len(val_str[digits_location + 1:])
            self.decimal_digits = 10 ** -n

    def getClassification(self):
        '''
        :return:返回种类及其个数
        '''
        return self.classification

    def setPriorProbability(self):
        '''
        :return:
        '''
        # setPy
        ck = list(self.classification.keys())
        ck.sort()
        sum_total = sum(self.classification.values())
        for key in ck:
            self.Py[key] = (self.indicatorYFunction(key) + 1) / (sum_total + self.characteristicQuantity)

        # setPxy
        index_ls = []
        Px = {}
        Pxx = {}
        for key in ck:
            for index, value in enumerate(self.y):
                if value == key:
                    index_ls.append(index)
            for j in range(self.characteristicQuantity):
                for val in np.arange(self.characteristicMinValue[j], self.characteristicMaxValue[j] + 1,
                                     self.decimal_digits / 10):
                    xy = self.indicatorXYFunction(val, j)
                    if xy != 0:
                        Px[val] = (xy + 1) / (self.indicatorYFunction(key) + 1 * self.xcharacteristicQuantity(j))
                cPx = copy.deepcopy(Px)
                Pxx[j] = cPx
                Px.clear()
            cPxx = copy.deepcopy(Pxx)
            self.Pxy[key] = cPxx
            Pxx.clear()

    def xcharacteristicQuantity(self, i):
        xx = []
        for x in self.X:
            xx.append(x[i])
        return len(list(set(xx)))

    def characteristicValueRange(self):
        '''
        :return:
        '''
        characteristicValue = []
        for i in range(self.characteristicQuantity):
            for x in self.X:
                characteristicValue.append(x[i])
            self.characteristicMaxValue.append(round(max(characteristicValue)))
            self.characteristicMinValue.append(round(min(characteristicValue)))
            characteristicValue.clear()
        return self.characteristicMinValue, self.characteristicMaxValue

    def indicatorYFunction(self, key):
        '''
        :return:
        '''
        sum_y = 0
        for y in self.y:
            if key == y:
                sum_y += 1
        return sum_y

    def indicatorXYFunction(self, value, j):
        '''
        :return:
        '''
        sum_xjl = 0
        for x in self.X:
            if x[j] == value:
                sum_xjl += 1
        return sum_xjl

    def predict(self, X):
        '''
        :return:
        '''
        sum_Pxy = 1
        ck = list(self.classification.keys())
        ck.sort()
        pred_y = []
        pred_result = {}
        for i, x in enumerate(X):
            for key in ck:
                for index, value in enumerate(x):
                    if self.Pxy[key][index].__contains__(value):
                        sum_Pxy *= self.Pxy[key][index][value]
                pred_result[key] = self.Py[key] * sum_Pxy
            self.pred_result[i] = copy.deepcopy(pred_result)
            pred_result.clear()
            max_prices = max(zip(self.pred_result[i].values(), self.pred_result[i].keys()))
            pred_y.append(max_prices[1])

        return pred_y


# nb = NaiveBayes([[1, 2], [2, 4], [3, 5]], [4, 1, 1])
# classification = nb.getClassification()
# nb.setPriorProbability()
# print(nb.characteristicMinValue, nb.characteristicMaxValue)
# print(nb.Py)
# print(nb.Pxy)
# x = nb.Pxy[1][0][1]
# print(nb.predict([[1, 2], [2, 4], [3, 5]]))
# print(nb.pred_result)
X, y = datasets.load_iris(return_X_y=True)
train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.1)
myNB = NaiveBayes(train_x, train_y)
myNB.setPriorProbability()
predict = myNB.predict(test_x)
print('predict\t', predict)
print('true\t', list(test_y))
print(accuracy_score(test_y, predict))

clf = GaussianNB()
clf = clf.fit(train_x, train_y)
y_pred = clf.predict(test_x)
print('predict\t', list(y_pred))
print('true\t', list(test_y))
print(accuracy_score(test_y, y_pred))
