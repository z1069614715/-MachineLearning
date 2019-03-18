# KNN算法

**numpy相关**

      import numpy as np
      a = np.array([1, 2, 3])
      print(np.tile(a, 2))  # 向左复制1次
      print(np.tile(a, (1, 2)))  # 先向左复制1次，但整体数组会变化，变成2维数组
      print(np.tile(a, (2, 2)))  # 先向左复制1次，再向下复制1次
      print(np.tile(a, (2, 2)).shape)  # (2,6)
      print(np.tile(a, (1, 2, 2)))  # 先向左复制1次，再向下复制1次，但整体数组会变化，变成3维数组
      print(np.tile(a, (1, 2, 2)).shape)  # (1,2,6)
