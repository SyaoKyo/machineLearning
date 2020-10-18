import copy
from collections import Counter

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from treelib import Tree, Node
from sklearn import datasets
import numpy as np
'''
自写kd树建立与最近邻搜索法
'''


def getDimData(X, dim):
    data = []
    for Xdata in X:
        data.append(Xdata[dim])
    return data


def getSplitDimData(X, index):
    data = []
    if not index:
        return data
    for i in index:
        data.append(X[i])
    return data


def split(X, dim):
    mdata = []
    Mdata = []
    eqdata = []
    data = getDimData(X, dim)
    median = np.median(data)
    eqdata.append(median)
    for i, v in enumerate(X):
        if v[dim] < median:
            mdata.append(v)
        elif v[dim] == median:
            eqdata.append(v)
        else:
            Mdata.append(v)
    return mdata, eqdata, Mdata


# 构建kd树
def createKDTree(X):
    tree = Tree()
    i = 0
    tree.create_node(1, 1, data=X)  # 初始化根节点
    maxNum = -1
    while True:
        i += 1
        if i == maxNum + 1:
            break
        if tree.get_node(i) is None:
            continue
        XX = tree.get_node(i).data
        if not len(XX):
            continue
        dim = tree.depth(i) % 4
        mdata, eqdata, Mdata = split(XX, dim)
        tree.get_node(i).data = eqdata  # 修改该结点
        tree.create_node(2 * i, 2 * i, parent=i, data=mdata)  # 左子树
        tree.create_node(2 * i + 1, 2 * i + 1, parent=i, data=Mdata)  # 右子树
        if not Mdata and not mdata and eqdata:
            if 2 * i + 1 > maxNum:
                maxNum = 2 * i + 1
    return tree


# 搜索kd树——最近邻搜索
def find(root, x, dim, i):
    dim = (dim + 1) % 4 - 1
    node = root.get_node(i)
    if not node.data:
        return tree.subtree(tree.ancestor(i))
    if len(node.data) == 1:
        if x[dim] < node.data[0]:
            return find(root.subtree(2 * i), x, dim, 2 * i)
        elif x[dim] > node.data[0]:
            return find(root.subtree(2 * i + 1), x, dim, 2 * i + 1)
        else:
            return root
    if x[dim] < node.data[1][dim]:
        if len(root.children(i)[1].data) == 1 or not root.children(i)[1].data:
            return root
        else:
            return find(root.subtree(2 * i), x, dim, 2 * i)
    elif x[dim] > node.data[1][dim]:
        if len(root.children(i)[1].data) == 1 or not root.children(i)[1].data:
            return root
        else:
            return find(root.subtree(2 * i + 1), x, dim, 2 * i + 1)
    else:
        return root


def predict(root, test_x):
    result = []
    res = []
    pred = []
    for x in test_x:
        ctree = find(root, x, 0, 1)
        node = ctree.all_nodes()
        data = node[0].data
        for d in data:
            for i, v in enumerate(train_x):
                if v.tolist() == d.tolist():
                    res.append(train_y[i])
        ress = copy.deepcopy(res)
        result.append(ress)
        res.clear()
    for re in result:
        pred.append(Counter(re).most_common(1)[0][0])
    return pred


'''
sklearn自带最近邻搜索法
'''
X, y = datasets.load_iris(return_X_y=True)

train_x, test_x, train_y, test_y = train_test_split(
    X, y, test_size=0.1,random_state=42)

#自写
tree = createKDTree(train_x)
tree.show()
pred = predict(tree.subtree(1), test_x)

#sklearn自带
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_x, train_y)
predict = knn.predict(test_x)

#评价
print(pred)
print(predict.tolist())
print(test_y.tolist())
print(accuracy_score(test_y, pred))
print(accuracy_score(test_y, predict))
