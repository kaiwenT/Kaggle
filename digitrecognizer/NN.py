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


def classify(test, train, label):
    mindis = 65535
    min = 0
    for v in train:
        dis = 0
        for i in range(0, len(v)):
            dis += math.pow(test[i] - v[i], 2)
        if math.sqrt(dis) < mindis:
            mindis = math.sqrt(dis)
            min = train.index(v)

    return label[min]


train, label = readtrainset('E:/Kaggle/DigitRecog/train1k.csv')
test = readtestset('E:/Kaggle/DigitRecog/test.csv')

f = open('E:/Kaggle/DigitRecog/submission1.csv', 'a')
f.write('ImageId,Label\n')
n = 1
for t in test:
    cls = classify(t, train, label)
    print(cls)
    f.write(str(n) + ',' + str(cls) + '\n')
    n += 1
f.close()
