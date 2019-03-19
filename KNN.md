# KNN算法

**numpy相关**

* numpy.tile()

      import numpy as np
      a = np.array([1, 2, 3])
      print(np.tile(a, 2))  # 向左复制1次
      print(np.tile(a, (1, 2)))  # 先向左复制1次，但整体数组会变化，变成2维数组
      print(np.tile(a, (2, 2)))  # 先向左复制1次，再向下复制1次
      print(np.tile(a, (2, 2)).shape)  # (2,6)
      print(np.tile(a, (1, 2, 2)))  # 先向左复制1次，再向下复制1次，但整体数组会变化，变成3维数组
      print(np.tile(a, (1, 2, 2)).shape)  # (1,2,6)

* numpy.sum()

      >>> np.sum([[0, 1], [0, 5]], axis=0) # 按列进行相加
      array([0, 6])
      >>> np.sum([[0, 1], [0, 5]], axis=1) # 按行进行相加
      array([1, 5])
      >>> np.sum([[0, 1], [0, 5]]) # 不加参数就全加
      6

* numpy.argsort()

      numpy.argsort(a, axis=-1, kind=’quicksort’, order=None) 
      功能: 将矩阵a按照axis排序，并返回排序后的下标 
      参数: a:输入矩阵， axis:需要排序的维度 
      返回值: 输出排序后的下标！！！！！
      >>> x = np.array([[1, 5, 7], [3, 2, 4]])
      >>> np.argsort(x, axis=0)
      array([[0, 1, 1],
             [1, 0, 0]])  #沿着行向下(每列)的元素进行排序
      >>> np.argsort(x, axis=1)
      array([[0, 1, 2],
             [1, 0, 2]])  #沿着列向右(每行)的元素进行排序

**K-近邻算法**
* 优点:精度高、对异常值不敏感、无数据输入假定.
* 缺点:计算复杂度高、空间复杂度高.
* 适用数据范围:数值型和标称型(真与假)

**KNN算法伪代码**
* 计算已知类别数据集中的点与当前点之间的距离;
* 按照距离递增次序排序;
* 选取与当前点距离最小的k个点;(通常k是不大于20的整数)
* 确定前k个点所在类别的出现频率;
* 返回前k个点出现频率最高的类别作为当前点的预测分类;

**KNN算法代码**

      import numpy as np
      import os

      def KNNSolve(inX, D, L, k):
          # 数据的行和列
          [m, n] = np.array(D).shape
          # 未知点与数据矩阵D内每个点的距离
          differeMat = np.tile(inX, (m, 1)) - D
          distanceSqur = (differeMat ** 2).sum(axis=1)
          distance = distanceSqur ** 0.5
          # 距离按照从小到大排序（前K个索引）
          indices = np.argsort(distance)[:k]
          # 参与投票的标签
          voteLabels = L[indices]
          # 标签个数统计字典
          count = {}
          for label in voteLabels:
              count[label] = count.get(label, 0) + 1  # 如果字典里有面有get后就+1没就添加进去再+1
          res = sorted(count.items(),key=lambda x:x[1], reverse=True)
          return res[0][0]
      path = os.getcwd() + "\\datingTestSet.txt"
      f = open(path)
      AllData = f.readlines()  # 只能执行一次,执行第二次的时候里面为空
      Data = []
      Label = []
      InX = np.array(AllData[0].rstrip('\n').split('\t')[:3]).astype(dtype=float)
      for line in AllData[1:]:
          line = line.rstrip('\n').split('\t')
          Data.append(line[:3])
          Label.append(line[3])
      print(KNNSolve(InX,np.array(Data).astype(dtype=float),np.array(Label),50))

**数据的归一化处理**

数据的归一化处理就是把数据的取值范围处理为0到1或者-1到1之间  
公式 : newValue = (oldValue - min) / (max - min) `这里的min和max指代的是每列数据的min值和max值`
