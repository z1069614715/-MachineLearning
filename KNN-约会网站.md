# 约会网站配对效果KNN算法代码

    import os
    import numpy as np


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
    DataSet = f.readlines()
    TestDataSet = [];TestLabelSet = []
    # 取出前面5个数据进行测试
    for i in DataSet[:5]:
        TestDataSet.append(list(map(float, i.rstrip('\n').split('\t')[:3])))
        TestLabelSet.append(i.rstrip('\n').split('\t')[3])
    Data = [];Label = []
    for i in DataSet[5:]:
        Data.append(list(map(float, i.rstrip('\n').split('\t')[:3])))
        Label.append(i.rstrip('\n').split('\t')[3])
    for i in range(len(TestDataSet)):
        print(KNN(np.array(TestDataSet[i]), np.array(Data), np.array(Label), 10) == TestLabelSet[i])

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
