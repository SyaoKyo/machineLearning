# project    :don't to try
# fileName   :NaiveBayes.py
# user       :cheng
# createDate :2020-11-12 16:12

import collections
import copy


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
        self.characteristicValueRange()

    def getClassification(self):
        '''
        :return:返回种类及其个数
        '''
        self.classification = collections.Counter(self.y)
        return self.classification

    def setPriorProbability(self):
        '''
        :return:
        '''
        ck = list(self.classification.keys())
        ck.sort()
        sum_total = sum(self.classification.values())
        for key in ck:
            self.Py[key] = self.indicatorYFunction(key) / sum_total

        index_ls = []
        Px = {}
        Pxx = {}
        for key in ck:
            for index, value in enumerate(self.y):
                if value == key:
                    index_ls.append(index)
            for j in range(self.characteristicQuantity):
                for val in range(self.characteristicMinValue[j], self.characteristicMaxValue[j] + 1):
                    if self.indicatorXYFunction(val, j) != 0:
                        Px[val] = self.indicatorXYFunction(val, j) / self.indicatorYFunction(key)
                cPx = copy.deepcopy(Px)
                Pxx[j] = cPx
                Px.clear()
            cPxx = copy.deepcopy(Pxx)
            self.Pxy[key] = cPxx
            Pxx.clear()

    def characteristicValueRange(self):
        '''
        :return:
        '''
        characteristicValue = []
        for i in range(self.characteristicQuantity):
            for x in self.X:
                characteristicValue.append(x[i])
            self.characteristicMaxValue.append(max(characteristicValue))
            self.characteristicMinValue.append(min(characteristicValue))
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
        for key in ck:
            for index, value in enumerate(X):
                print(key,index,value)
                if self.Pxy[key][index].__contains__(value):
                    sum_Pxy *= self.Pxy[key][index][value]
            self.pred_result[key] = self.Py[key] * sum_Pxy

        return [k for k, v in self.pred_result.items() if v == max(self.pred_result.values())]


nb = NaiveBayes([[1, 2], [2, 4], [3, 5]], [4, 1, 1])
classification = nb.getClassification()
nb.setPriorProbability()
print(nb.characteristicMinValue, nb.characteristicMaxValue)
print(nb.Py)
print(nb.Pxy)
x = nb.Pxy[1][0][1]
print(nb.predict([10,31]))
print(nb.pred_result)

