

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
