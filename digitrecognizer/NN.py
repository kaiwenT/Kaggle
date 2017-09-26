import math
import tools


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


train, label = tools.readtrainset('E:/Kaggle/DigitRecog/train1k.csv')
test = tools.readtestset('E:/Kaggle/DigitRecog/test.csv')

f = open('E:/Kaggle/DigitRecog/submission1.csv', 'a')
f.write('ImageId,Label\n')
n = 1
for t in test:
    cls = classify(t, train, label)
    print(cls)
    f.write(str(n) + ',' + str(cls) + '\n')
    n += 1
f.close()
