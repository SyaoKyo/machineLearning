import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class myPerceptron(object):
    def __init__(self):
        self.learning_step = 0.00001
        self.max_iteration = 5000

    def predict_(self, x):
        wx = sum([self.w[j] * x[j] for j in range(len(self.w))])
        return int(wx > 0)

    def train(self, features, labels):
        self.w = [0.0] * (len(features[0]) + 1)  # 权重+偏置
        correct_count = 0
        time = 0

        while time < self.max_iteration:
            index = np.random.randint(0, len(labels) - 1)  # 随机找个样本
            x = list(features[index])
            x.append(1.0)  # 偏置
            y = 2 * labels[index] - 1
            wx = sum([self.w[j] * x[j] for j in range(len(self.w))])

            if wx * y > 0:
                correct_count += 1
                if correct_count > self.max_iteration:
                    break
                continue

            # 错误就更新参数
            for i in range(len(self.w)-1):
                self.w[i] += self.learning_step * (y * x[i])
            self.w[-1] += self.learning_step * y

    def predict(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_(x))
        return labels


if __name__ == '__main__':
    # 读取数据
    X, y = datasets.load_breast_cancer(return_X_y=True)
    # PCA降维
    pca = PCA(n_components=2)
    pca.fit(X)
    X = pca.fit_transform(X)
    # 标准化
    sc = StandardScaler()
    sc.fit(X)
    X = sc.transform(X)
    X = sc.transform(X)
    # 标明正例和反例
    positive_x1 = [X[i, 0] * 1000 for i in range(len(X)) if y[i] == 1]
    positive_x2 = [X[i, 1] * 1000 for i in range(len(X)) if y[i] == 1]
    negetive_x1 = [X[i, 0] * 1000 for i in range(len(X)) if y[i] == 0]
    negetive_x2 = [X[i, 1] * 1000 for i in range(len(X)) if y[i] == 0]

    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        X, y, test_size=0.33, random_state=0)

    # 自写感知机
    p = myPerceptron()
    p.train(train_features, train_labels)
    mw = -p.w[0] / p.w[1]
    mb = p.w[2] / p.w[1]
    test_predict = p.predict(test_features)
    score = accuracy_score(test_labels, test_predict)

    # sklearn自带感知机
    p = Perceptron()
    p.fit(train_features, train_labels)
    w = -p.coef_[0][0] / p.coef_[0][1]
    b = p.intercept_ / p.coef_[0][1]
    predy = p.predict(test_features)

    # 模型准确率
    print("myPerceptron's accruacy socre is ", score)
    print("Perceptron's accruacy socre is ", accuracy_score(test_labels, predy))

    # 作图
    line_x = [-1, 1]
    line_y = [line_x[0] * mw - mb,
              line_x[1] * mw - mb]

    line_x1 = [-1, 1]
    line_y1 = [line_x[0] * w - b,
               line_x[1] * w - b]
    plt.scatter(positive_x1, positive_x2, 5, 'r')
    plt.scatter(negetive_x1, negetive_x2, 5, 'g')
    plt.plot(line_x, line_y, 'b')
    plt.plot(line_x1, line_y1, 'm')
    plt.legend(['myPerceptron', 'Perceptron'])
    plt.show()
