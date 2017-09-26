from math import sqrt
from collections import namedtuple
import tools


class kdNode(object):
    def __init__(self, vector, split, left, right):
        self.vector = vector
        self.split = split
        self.left = left
        self.right = right


# 创建KD树
class kdTree(object):
    def __init__(self, data):
        k = len(data[0])

        def createNode(split, dataset):
            if not dataset:
                return None

            dataset.sort(key=lambda x : x[split])
            split_pos = len(dataset) // 2
            medium = dataset[split_pos]
            split_next = (split_pos + 1) % k

            return kdNode(medium, split,
                          createNode(split_next, dataset[:split_pos]),
                          createNode(split_next, dataset[split_pos + 1:]))

        self.root = createNode(0, data)

res = namedtuple("result_tuple", "nearest_point nearest_dist nodes_visited")


# 搜索KD树
def find_nearest(tree, point):
    k = len(point)

    def travel(kdnode, target, max_dist):
        if kdnode is None:
            return res([0] * k, float("inf"), 0)

        nodes_visited = 1

        s = kdnode.split
        pivot = kdnode.vector

        if target[s] <= pivot[s]:
            nearer_node = kdnode.left
            further_node = kdnode.right
        else:
            nearer_node = kdnode.right
            further_node = kdnode.left

        temp1 = travel(nearer_node, target, max_dist)

        nearest = temp1.nearest_point
        dist = temp1.nearest_dist

        nodes_visited += temp1.nodes_visited

        if dist < max_dist:
            max_dist = dist

        temp_dist = abs(pivot[s] - target[s])
        if max_dist < temp_dist:
            return res(nearest, dist, nodes_visited)

        temp_dist = sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(pivot, target)))

        if temp_dist < dist:
            nearest = pivot
            dist = temp_dist
            max_dist = dist

        temp2 = travel(further_node, target, max_dist)

        nodes_visited += temp1.nodes_visited

        if temp2.nearest_dist < dist:
            nearest = temp2.nearest_point
            dist = temp2.nearest_dist

        return res(nearest, dist, nodes_visited)

    return travel(tree.root, point, float("inf"))


if __name__ == "__main__":
    train, label = tools.readtrainset('E:/Kaggle/DigitRecog/train.csv')
    test = tools.readtestset('E:/Kaggle/DigitRecog/test.csv')

    f = open('E:/Kaggle/DigitRecog/submission_kd.csv', 'a')
    f.write('ImageId,Label\n')
    n = 1
    kd = kdTree(train)

    for t in test:
        ret = find_nearest(kd, t)

        cls = label[train.index(ret.nearest_point)]
        print(cls)
        f.write(str(n) + ',' + str(cls) + '\n')
        n += 1
    f.close()
