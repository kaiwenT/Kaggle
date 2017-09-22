import math
from operator import itemgetter

def readtrainset(filename):
    f = open(filename, 'r')
    trains = []
    label = []
    next(f)
    for line in f:
        strs = line.split(',')
        nums = [int(x) for x in strs]
        label.append(nums[0])
        trains.append(nums[1:])
    f.close()
    return trains, label


def readtestset(filename):
    f = open(filename, 'r')
    test = []

    next(f)
    for line in f:
        strs = line.split(',')
        nums = [int(x) for x in strs]
        test.append(nums)
    f.close()
    return test


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


train, label = readtrainset('E:/Kaggle/DigitRecog/train1k.csv')
test = readtestset('E:/Kaggle/DigitRecog/test.csv')

f = open('E:/Kaggle/DigitRecog/submission1.csv', 'a')
f.write('ImageId,Label\n')
n = 1
for t in test:
    cls = classify(t, train, label, 20)
    print(cls)
    f.write(str(n) + ',' + str(cls) + '\n')
    n += 1
f.close()
