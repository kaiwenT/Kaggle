import math
from operator import itemgetter
import tools


def classify(test, train, label, k):
    diff = dict()
    for v in train:
        dis = 0
        for i in range(0, len(v)):
            dis += math.pow(test[i] - v[i], 2)
        diff[train.index(v)] = math.sqrt(dis)

    number = dict()
    for i in range(0, 10):
        number[i] = 0
    for k, v in sorted(diff.items(), key=itemgetter(1), reverse=True)[0:k]:
        number[label[k]] += 1

    for k, v in sorted(number.items(), key=itemgetter(1), reverse=True):
        return k


train, label = tools.readtrainset('E:/Kaggle/DigitRecog/train1k.csv')
test = tools.readtestset('E:/Kaggle/DigitRecog/test.csv')

f = open('E:/Kaggle/DigitRecog/submission1.csv', 'a')
f.write('ImageId,Label\n')
n = 1
for t in test:
    cls = classify(t, train, label, 20)
    print(cls)
    f.write(str(n) + ',' + str(cls) + '\n')
    n += 1
f.close()
