# 感知机（Perceptron）

## 感知机模型

感知机定义：假设输入空间（特征空间）是 ![](https://latex.codecogs.com/gif.latex?\\\mathcal{X}\in%20R^n)，输出空间是![](http://latex.codecogs.com/gif.latex?\\\mathcal{Y}=\lbrace+1,-1\rbrace),输入![](http://latex.codecogs.com/gif.latex?\\x\in\mathcal{X})表示实例的特征向量，对应于输入空间（特征空间）的点；输出![](http://latex.codecogs.com/gif.latex?\\y\in\mathcal{Y})表示实例的类别，由输入空间到输出空间的模型称为感知机。如下函数：
![](http://latex.codecogs.com/gif.latex?\\f(x)=sign(w \cdot x + b))
其中，![](http://latex.codecogs.com/gif.latex?\\w)和![](http://latex.codecogs.com/gif.latex?\\b)为感知机模型参数，![](http://latex.codecogs.com/gif.latex?\\w\in%20R^n)叫作权值（weight）或权值向量（weight vector），![](http://latex.codecogs.com/gif.latex?\\b\in%20R)叫作偏置（bias），![](http://latex.codecogs.com/gif.latex?\\w\cdot%20x)表示![](http://latex.codecogs.com/gif.latex?\\w)和![](http://latex.codecogs.com/gif.latex?\\x)的内积。sign是符号函数，即

![](https://latex.codecogs.com/gif.latex?\\%20sign(x)=\left\{%20\begin{array}{rcl}%20+1,%20&%20x\geq%200\\%20-1,%20&%20x%3C0\\%20\end{array}%20\right.)


感知机是一种线性分类模型，属于判别模型。感知机模型的假设空间是定义在特征空间中的所有线性分类模型（linear classification model）或线性分类器（linear classifier），即函数集合![](https://latex.codecogs.com/gif.latex?\\\{f|f(x)=w%20\cdot%20x%20+%20b\})。

感知机有如下几何解释

![](https://latex.codecogs.com/gif.latex?\\%20w%20\cdot%20x%20+%20b%20=%200)

对应于特征空间![](https://latex.codecogs.com/gif.latex?\\R^n)中的一个超平面![](http://latex.codecogs.com/gif.latex?\\S)，其中![](http://latex.codecogs.com/gif.latex?\\w)是超平面的法向量，![](http://latex.codecogs.com/gif.latex?\\b)是超平面的截距。这个超平面将特征空间划分为两个部分。位于两部分的点（特征向量）分别被分为正、负两类。因此，超平面![](http://latex.codecogs.com/gif.latex?\\S)称为分离超平面（separating hyperplane），如下图所示。

![](.\图片\感知机\感知机模型.jpg)

感知机学习，由训练数据集（市里的特征向量及类别）
![](https://latex.codecogs.com/gif.latex?\\T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\})
其中，![](https://latex.codecogs.com/gif.latex?\\x_i\in\mathcal{X}=R^n,y_i\in\mathcal{Y}=\{+1,-1\},i=%201,2,\cdots,N),求得感知机模型，即求得模型参数![](http://latex.codecogs.com/gif.latex?\\w,b)。感知机预测，通过学习得到的感知机模型，对于新的输入实例给出其对应的输出类别。

## 感知机学习策略

### 数据集的线性可分性

给定一个数据集
![](https://latex.codecogs.com/gif.latex?\\T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\})
其中，![](https://latex.codecogs.com/gif.latex?\\x_i\in\mathcal{X}=R^n,y_i\in\mathcal{Y}=\{+1,-1\},i=%201,2,\cdots,N),如果存在某个超平面S
![](https://latex.codecogs.com/gif.latex?\\%20w%20\cdot%20x%20+%20b%20=%200)
能够将数据集的正实例点和负实例点完全正确的划分到超平面的两侧，即对所有![](http://latex.codecogs.com/gif.latex?\\y_i=+1)的实例![](http://latex.codecogs.com/gif.latex?\\i)，有![](https://latex.codecogs.com/gif.latex?\\w%20\cdot%20x%20+%20b%20%3E%200)，对所有![](http://latex.codecogs.com/gif.latex?\\y_i=-1)的实例![](http://latex.codecogs.com/gif.latex?\\i)，有![](https://latex.codecogs.com/gif.latex?\\w%20\cdot%20x%20+%20b%20%3C%200)，则称数据集T为线性可分数据集（linearly separable data set）；否则，称数据集T线性不可分。

------

### 感知机学习策略

输入空间![](http://latex.codecogs.com/gif.latex?\\R^n)中的任一点![](http://latex.codecogs.com/gif.latex?\\x_0)到超平面![](http://latex.codecogs.com/gif.latex?\\S)的距离：

![](https://latex.codecogs.com/gif.latex?\\%20\frac{1}{\|w\|}|w%20\cdot%20x_0%20+%20b|)


其中![](http://latex.codecogs.com/gif.latex?\\\|w\|)是![](http://latex.codecogs.com/gif.latex?\\w)的![](http://latex.codecogs.com/gif.latex?\\L_2)范数。

对于误分类数据![](http://latex.codecogs.com/gif.latex?\\(x_i,y_i))，有
![](https://latex.codecogs.com/gif.latex?\\%20-y_i(w%20\cdot%20x%20+%20b)%20%3E%200)
成立。因为当 ![](http://latex.codecogs.com/gif.latex?\\w \cdot x + b > 0) 时，![](http://latex.codecogs.com/gif.latex?\\y_i=-1) ，当![](https://latex.codecogs.com/gif.latex?\\w%20\cdot%20x%20+%20b%20%3C%200) 时，![](http://latex.codecogs.com/gif.latex?\\y_i=+1)。所以误分类点 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D) 到分离超平面的距离:

![](https://latex.codecogs.com/gif.latex?\\%20-\frac{1}{\|w\|}y_i(w%20\cdot%20x_i%20+%20b))

假设超平面![](http://latex.codecogs.com/gif.latex?\\S)的误分类点集合为![](http://latex.codecogs.com/gif.latex?\\M)，则所有误分类点到超平面![](http://latex.codecogs.com/gif.latex?\\S)的总距离：

![](http://latex.codecogs.com/gif.latex?\\%20-\frac{1}{\|w\|}\sum_{x_i%20\in%20M}y_i(w%20\cdot%20x_i%20+%20b))
不考虑![](http://latex.codecogs.com/gif.latex?\\\frac{1}{\|w\|})，就能得到感知机学习的损失函数。给定训练数据集
![](https://latex.codecogs.com/gif.latex?\\T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\})
其中，![](https://latex.codecogs.com/gif.latex?\\x_i\in\mathcal{X}=R^n,y_i\in\mathcal{Y}=\{+1,-1\},i=%201,2,\cdots,N)。感知机![](https://latex.codecogs.com/gif.latex?\\sign(w%20\cdot%20x%20+%20b))的损失函数定义为
![](https://latex.codecogs.com/gif.latex?\\%20L(w,b)=%20-%20\sum_{x_i%20\in%20M}y_i(w%20\cdot%20x_i%20+%20b))
其中，![](http://latex.codecogs.com/gif.latex?\\M)为误分类点的集合。这个损失函数就是感知机学习的经验风险函数。

显然，损失函数![](http://latex.codecogs.com/gif.latex?\\L(w,b))是非负的。如果没有误分类点，损失函数值是0。而且，误分类点越少，误分类点离超平面越近，损失函数值就越小。其中，损失函数![](http://latex.codecogs.com/gif.latex?\\L(w,b))是![](http://latex.codecogs.com/gif.latex?\\w,b)的连续可导函数。

------

### 感知机学习算法

#### 算法的原始形式

感知机学习算法是对以下最优化问题的算法。给定训练数据集
![](https://latex.codecogs.com/gif.latex?\\T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\})
其中，![](https://latex.codecogs.com/gif.latex?\\x_i\in\mathcal{X}=R^n,y_i\in\mathcal{Y}=\{+1,-1\},i=%201,2,\cdots,N)。感知机![](https://latex.codecogs.com/gif.latex?\\sign(w%20\cdot%20x%20+%20b))的损失函数定义为
![](https://latex.codecogs.com/gif.latex?\\%20\min_{w,b}L(w,b)=%20-%20\sum_{x_i%20\in%20M}y_i(w%20\cdot%20x_i%20+%20b))
其中，![](http://latex.codecogs.com/gif.latex?\\M)为误分类点的集合。感知机学习算法是误分类驱动的，具体采用随机梯度下降法（stochastic gradient descent）。首先，任意选取一个超平面![](http://latex.codecogs.com/gif.latex?\\w_0,b_0)，然后用梯度下降法不断地极小化目标函数![](http://latex.codecogs.com/gif.latex?\\\min_{w,b}L(w,b)= - \sum_{x_i \in M}y_i(w \cdot x_i + b))。极小化过程中不是一次使![](http://latex.codecogs.com/gif.latex?\\M)中所有误分类点的梯度下降，而是一次随机选取一个误分类点使其梯度下降。

假设误分类点集合![](http://latex.codecogs.com/gif.latex?\\M)是固定的，那么损失函数![](http://latex.codecogs.com/gif.latex?\\L(w,b))的梯度由
![](https://latex.codecogs.com/gif.latex?\\%20\triangledown_w%20L(w,b)=-%20\sum_{x_i%20\in%20M}y_ix_i\\%20\triangledown_b%20L(w,b)=-%20\sum_{x_i%20\in%20M}y_i)
给出。

随机选取一个误分类点![](http://latex.codecogs.com/gif.latex?\\(x_i,y_i))，对![](http://latex.codecogs.com/gif.latex?\\w,b)进行更新：
![](https://latex.codecogs.com/gif.latex?\\%20w%20\leftarrow%20w%20+%20\eta%20y_ix_i%20\\%20b%20\leftarrow%20b%20+%20\eta%20y_i)
式中![](https://latex.codecogs.com/gif.latex?\\\eta(0%20%3C%20\eta%20\leq%201))是步长，在统计学习中又称为学习率（learning rate）。这样，通过迭代可以期待损失函数![](http://latex.codecogs.com/gif.latex?\\L(w,b))不断减小，直到0.综上所述，得到如下算法：

感知机学习算法（原始形式）

> 输入：训练数据集![](http://latex.codecogs.com/gif.latex?\\T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\})其中，![](http://latex.codecogs.com/gif.latex?\\x_i\in\mathcal{X}=R^n)，![](https://latex.codecogs.com/gif.latex?\\y_i\in\mathcal{Y}=\{-1,1\})，![](https://latex.codecogs.com/gif.latex?\\i=%201,2,\cdots,N)；学习率![](https://latex.codecogs.com/gif.latex?\\\eta(0%20%3C%20\eta%20\leq%201))；
>
> 输出：![](http://latex.codecogs.com/gif.latex?\\w,b)；感知机模型![](https://latex.codecogs.com/gif.latex?\\f(x)=sign(w%20\cdot%20x%20+%20b))。
>
> （1）选取初值![](http://latex.codecogs.com/gif.latex?\\w_0,b_0);
>
> （2）在训练集中选取数据![](http://latex.codecogs.com/gif.latex?\\(x_i,y_i))；
>
> （3）如果![](https://latex.codecogs.com/gif.latex?\\y_i(w%20\cdot%20x_i%20+%20b)%20\leq%200),
> ![](https://latex.codecogs.com/gif.latex?\\%20%20w%20\leftarrow%20w%20+%20\eta%20y_ix_i%20\\%20%20b%20\leftarrow%20b%20+%20\eta%20y_i)
> （4）转至（2），直到训练集中没有误分类点。

这个算法是感知机学习的基本算法，对应于后面的对偶形式，称为原始形式。感知机学习算法简单且易于实现。

实现代码（自写+sklearn自带）：[详细代码](.\代码\myPerceptron.py)

```python
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
        X, y, test_size=0.33, random_state=23323)

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

```

划分结果如下：

![](.\图片\感知机\自制练习.png)



>模型准确率：
>
>myPerceptron's accruacy socre is  0.9468085106382979
>
>Perceptron's accruacy socre is  0.9574468085106383

可见自写感知机算法正确，可以实现二分类操作。

#### 算法收敛性

**定理.Novikoff** 设训练数据集![](http://latex.codecogs.com/gif.latex?\\T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\})是线性可分的， 其中，![](http://latex.codecogs.com/gif.latex?\\x_i\in\mathcal{X}=R^n)，![](https://latex.codecogs.com/gif.latex?\\y_i\in\mathcal{Y}=\{-1,1\})，![](https://latex.codecogs.com/gif.latex?\\i=%201,2,\cdots,N)，则：

（1）存在满足条件  ![](http://latex.codecogs.com/gif.latex?\\\|\hat{w}_{opt}\|=1)的超平面  ![](https://latex.codecogs.com/gif.latex?\\\hat{w}_{opt}\cdot%20\hat{x}=w_{opt}x%20+%20b_{opt}%20=%200)将训练数据集完整正确分开；并存在![](https://latex.codecogs.com/gif.latex?\\\gamma%20%3E%200)，对所有 ![](https://latex.codecogs.com/gif.latex?\\i=%201,2,\cdots,N)
![](https://latex.codecogs.com/gif.latex?\\%20y_i(\hat{w}_{opt}\cdot%20\hat{x}_i)=y_i(w_{opt}x_i%20+%20b_{opt})\geq\gamma)
（2）令 ![](https://latex.codecogs.com/gif.latex?\\R%20=%20\max%20\limits_{1\leq%20i%20\leq%20N}\|\hat{x}_i\|)，则感知机算法在训练集上的误分类次数 ![](http://latex.codecogs.com/gif.latex?\\k)满足不等式：
![](https://latex.codecogs.com/gif.latex?\\%20k%20\leq%20(\frac{R}{\gamma})^2)
定理标明，误分类的次数![](http://latex.codecogs.com/gif.latex?\\k)是有上界的，经过有限次搜索可以找到将训练数据完全正确分开的分离超平面。也就是说，当训练数据集可分时，感知机学习算法原始形式迭代是收敛的。

注：感知机学习算法存在许多解，这些解既依赖于初值的选择，也依赖于迭代过程中误分类点的选择顺序。为了得到唯一的超平面，需要对分离超平面增加约束条件。这就是线性支持向量机的想法。当训练集线性不可分时，感知机学习算法不收敛，迭代结果会发生震荡。

#### 算法对偶形式

对偶形式的基本想法是，将![](http://latex.codecogs.com/gif.latex?\\w)和![](http://latex.codecogs.com/gif.latex?\\b)表示为实例![](http://latex.codecogs.com/gif.latex?\\x_i)和标记![](http://latex.codecogs.com/gif.latex?\\y_i)的线性组合的形式，通过求解其系数而求得![](http://latex.codecogs.com/gif.latex?\\w)和![](http://latex.codecogs.com/gif.latex?\\b)。不失一般性，在原始形式中可假设初始值![](http://latex.codecogs.com/gif.latex?\\w_0,b_0)均为0.对误分类点![](http://latex.codecogs.com/gif.latex?\\(x_i,y_i))通过

![](https://latex.codecogs.com/gif.latex?\\%20w%20\leftarrow%20w%20+%20\eta%20y_ix_i%20\\%20b%20\leftarrow%20b%20+%20\eta%20y_i)
逐步修改![](http://latex.codecogs.com/gif.latex?\\w,b)，设修改n次，则，![](http://latex.codecogs.com/gif.latex?\\w,b)关于![](http://latex.codecogs.com/gif.latex?\\(x_i,y_i))的增量分别是![](http://latex.codecogs.com/gif.latex?\\a_iy_ix_i)和![](http://latex.codecogs.com/gif.latex?\\a_iy_i)，这里![](http://latex.codecogs.com/gif.latex?\\a_i=n_i\eta)。这样，从学习过程中不难看出，最后学习到的![](http://latex.codecogs.com/gif.latex?\\w,b)可以分别表示为
![](https://latex.codecogs.com/gif.latex?\\w=\sum^{N}_{i=1}a_iy_ix_i\\b=\sum^{N}_{i=1}a_iy_i)
这里，![](http://latex.codecogs.com/gif.latex?\\a_i\geq0,i=1,2,\cdots,N)当![](http://latex.codecogs.com/gif.latex?\\\eta=1)，表示第![](http://latex.codecogs.com/gif.latex?\\i)实例点由于误分而进行更新的次数。实例点更新次数越多，意味着他距离分离超平面越近，也就越难正确分类。换句话说，这样的实例对学习结果影响最大。

下面对照原始形式来叙述感知机学习算法的对偶形式。

> 输入：训练数据集![](http://latex.codecogs.com/gif.latex?\\T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\})其中，![](http://latex.codecogs.com/gif.latex?\\x_i\in\mathcal{X}=R^n)，![](http://latex.codecogs.com/gif.latex?\\y_i\in\mathcal{Y}=\{-1,1\})，![](https://latex.codecogs.com/gif.latex?\\i=%201,2,\cdots,N)；学习率![](https://latex.codecogs.com/gif.latex?\\\eta(0%20%3C%20\eta%20\leq%201))；
>
> 输出：![](http://latex.codecogs.com/gif.latex?\\a,b)；感知机模型![](https://latex.codecogs.com/gif.latex?\\f(x)=sign(\sum\limits^{N}_{i=1}a_jy_jx_j\cdot%20x%20+%20b))，其中![](http://latex.codecogs.com/gif.latex?\\a=(a_1,a_2,\cdots,a_N)^T)。
>
> （1）![](http://latex.codecogs.com/gif.latex?\\a\leftarrow0,b\leftarrow0);
>
> （2）在训练集中选取数据![](http://latex.codecogs.com/gif.latex?\\(x_i,y_i))；
>
> （3）如果![](https://latex.codecogs.com/gif.latex?\\y_i(\sum\limits^{N}_{i=1}a_jy_jx_j\cdot%20x%20+%20b)%20\leq%200),
> ![](https://latex.codecogs.com/gif.latex?\\%20a_i%20\leftarrow%20a_i%20+%20\eta%20\\%20%20b%20\leftarrow%20b%20+%20\eta%20y_i)
> 		（4）转至（2），直到训练集中没有误分类点。

对偶性式中训练实例仅以内积的形式出现。为了方便，可以预先将训练集中实例间的内积计算出来并以矩阵的形式存储，这个矩阵就是所谓的Gram矩阵
![](https://latex.codecogs.com/gif.latex?\\%20G=[x_i\cdot%20x_j]_{N\times%20N})

## 总结

1. 感知机是根据输入实例的特征向量![](http://latex.codecogs.com/gif.latex?\\x)对其进行二类分类的线性分类模型：
   ![](https://latex.codecogs.com/gif.latex?\\%20f(x)=sign(w%20\cdot%20x%20+%20b))
   感知机模型对应于输入空间（特征空间）中的分离超平面![](https://latex.codecogs.com/gif.latex?\\w%20\cdot%20x%20+%20b=0)
   

   
2. 感知机学习的策略是极小化损失函数：
   ![](https://latex.codecogs.com/gif.latex?\\%20\min_{w,b}L(w,b)=%20-%20\sum_{x_i%20\in%20M}y_i(w%20\cdot%20x_i%20+%20b))
   损失函数对应于误分类点到分离超平面的总距离。
   

   
3. 感知机学习算法是基于随机梯度下降法的对损失函数的最优化算法，有原始形式和对偶形式。算法简单且易于实现。原始形式中，首先任意选取一个超平面，然后用梯度下降法不断极小化目标函数。在这个过程中一次随机选取一个误分类点使其梯度下降。

   

4. 当训练数据集线性可分时，感知机学习算法是收敛的。感知机算法在训练数据集上的误分类次数![](http://latex.codecogs.com/gif.latex?\\k)满足不等式：
   ![](https://latex.codecogs.com/gif.latex?\\%20k%20\leq%20(\frac{R}{\gamma})^2)
   当训练数据集线性可分时，感知机学习算法存在无穷多个解，其解由于不同的初值或不同的迭代顺序而可能有所不同。


