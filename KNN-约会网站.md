# 约会网站配对效果KNN算法代码

    import os
    import numpy as np


    def autoNorm(D):
        minv = D.min(0)
        maxv = D.max(0)
        n = D.shape[0]
        ranges = maxv - minv
        newValue = (D - np.tile(minv, (n, 1))) / (np.tile(ranges, (n, 1)))
        return newValue, minv, ranges


    def KNN(Inx, D, L, k):
        [m, n] = D.shape
        Different = ((np.tile(Inx, (m, 1)) - D) ** 2).sum(axis=1) ** 0.5
        DifferentIndex = np.argsort(Different)[:k]
        DifferentArr = L[DifferentIndex]
        LabelCount = {}
        for i in DifferentArr:
            LabelCount[i] = LabelCount.get(i, 0) + 1
        res = sorted(LabelCount.items(), key=lambda x: x[1], reverse=True)
        return res[0][0]


    path = os.getcwd() + '\\datingTestSet2.txt'
    f = open(path)
    DataSet = []
    LabelSet = []
    for line in f.readlines():
        line = list(map(float, line.rstrip('\n').split('\t')))
        DataSet.append(line[:3])
        LabelSet.append(int(line[3]))
    DataSet, minv, ranges = autoNorm(np.array(DataSet))
    TestDataSet = DataSet[:100];TestLabelSet = LabelSet[:100];Data = DataSet[:];Label = LabelSet[:]
    # 使用数据集的一部分进行测试KNN算法的错误率
    falseTotal = 0
    for i in range(len(TestDataSet)):
        preRes = KNN(np.array(TestDataSet[i]), np.array(Data), np.array(Label), 5)
        if preRes != TestLabelSet[i]:
            falseTotal += 1
        print('the classifier came back with: {:d}, thr real answer is: {:d}'.format(preRes, TestLabelSet[i]))
    print('the total error rate is: {:.5f}'.format(falseTotal / len(TestDataSet)))
    # 通过输入数据来测试海伦对对方的喜欢程度
    res = ['not at all', 'in small doses', 'in large deses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    InArr = np.array([ffMiles, percentTats, iceCream])
    print('You will probably like this persion:',res[KNN((InArr - minv) / ranges, np.array(Data), np.array(Label), 10)])


# 约会网站散点图代码

    import numpy as np
    import matplotlib.pyplot as plt
    import os
    path = os.getcwd() + '\\datingTestSet2.txt'
    f = open(path)
    DataSet = []
    LabelSet = []
    for line in f.readlines():
        line = list(map(float,line.rstrip('\n').split('\t')))
        DataSet.append(line[:3])
        LabelSet.append(int(line[3]))
    DataSet = np.array(DataSet)
    LabelSet = np.array(LabelSet)
    Legend = ['不喜欢的人','魅力一般的人','极具魅力的人']
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(1,4):
        ax.scatter(DataSet[np.where(LabelSet == i),0],DataSet[np.where(LabelSet == i),1],label = Legend[i - 1])
    plt.legend() #必须加,不加不显示标签
    plt.xlabel('每年获取的飞行常客里程数')
    plt.ylabel('玩视频游戏所耗时间百分比')
    plt.rcParams['font.sans-serif']=['SimHei'] #用于正常显示中文
    plt.show()
