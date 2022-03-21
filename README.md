# handwritten_numbers_classifying
## 任务目标
设计一个算法实现对手写数字的识别。
## 任务流程
- 包的导入
- 设置超参数
- 预处理数据
- 训练神经网络模型
- 测试神经网络模型
## 包的导入
```
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cnn  # 自己建立的cnn类,用来实例化。
```
我们使用csv的包来提取储存于csv格式文件中的数据。  
使用numpy来做数据的初步处理。  
使用pytorch作为构建cnn的工具包。  
