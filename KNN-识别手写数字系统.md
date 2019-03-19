# KNN-识别手写数字系统

    import os
    import numpy as np

    # 转换成一维数组便于计算
    def imgSolve(filename):
        f = open(filename)
        returnVec = np.zeros((1, 1024))
        for i in range(32):
            line = f.readline()
            for j in range(32):
                returnVec[0, 32 * i + j] = int(line[j])
        return returnVec

    
    def KNNSolve(InX, D, L, k):
        n = D.shape[0]
        DisArr = ((np.tile(InX, (n, 1)) - D) ** 2).sum(axis=1) ** 0.5
        kIndex = np.argsort(DisArr)[:k]
        kArr = L[kIndex]
        LabelCount = {}
        for label in kArr:
            LabelCount[label] = LabelCount.get(label, 0) + 1
        dictArr = sorted(LabelCount.items(), key=lambda x: x[1], reverse=True)
        return dictArr[0][0]


    # 导入数据到DataSet和LabelSet
    path = os.getcwd() + '\\trainingDigits\\'
    trainingDigitsFileList = os.listdir(path)
    DataSet = np.zeros((len(trainingDigitsFileList), 1024))
    LabelSet = np.zeros(len(trainingDigitsFileList))
    for index, fileName in enumerate(trainingDigitsFileList):
        DataSet[index] = imgSolve(path + fileName)
        LabelSet[index] = fileName.split('_')[0]
    # 导入测试数据
    testPath = os.getcwd() + '\\testDigits\\'
    testDigits = os.listdir(testPath)
    # 测试KNN算法的错误率
    falseTotal = 0
    for index, fileName in enumerate(testDigits):
        testLabel = int(KNNSolve(imgSolve(testPath + fileName), DataSet, LabelSet, 3))
        label = int(fileName.split('_')[0])
        if testLabel != label:
            falseTotal += 1
        print('the classifier came back whit: {}, the real answer is: {}'.format(testLabel, label))
    print('\nthe total number of errors is: {}'.format(falseTotal))
    print('the total error rate is: {:.5f}'.format(falseTotal / len(testDigits)))
