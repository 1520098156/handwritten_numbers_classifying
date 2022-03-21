# handwritten_numbers_classifying
## 任务目标
设计一个CNN算法实现对手写数字的识别。
![image](https://user-images.githubusercontent.com/52622948/159292437-a1a67759-fd54-4d2c-865f-d25aee0d55a0.png)
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
## 设置超参数
在机器学习的上下文中，超参数是在开始学习过程之前设置值的参数。 相反，其他参数的值通过训练得出。  
```
# 定义超参数
input_size = 28  # 图像的总尺寸28*28
num_classes = 10  # 标签的种类数
num_epochs = 3  # 训练的总循环周期
batch_size = 64  # 一个撮（批次）的大小，64张图片
```
我们知道这次的数据集的手写数字图片数据尺寸是28\*28\*1的。  
0到9共计10个数字，因此我们需要10个分类标签。
我们设定设定三个epoch，这意味着我们将会把训练集循环训练三次。  
我们设定一个batch训练64张图片，这样可以防止同时训练过多数据导致内存不够。
## 预处理数据
```
def read_train_data(filename):
    data = []
    csv_reader = csv.reader(open(filename))
    next(csv_reader)
    for row in csv_reader:
        label = int(row[0])
        pic_in_row = [float(x) / 255 for x in row[1:]]
        pic = np.array(pic_in_row).reshape(1, 28, 28)
        pic = torch.from_numpy(pic).to(torch.float32)
        temp = (pic, label)
        data.append(temp)
    return data
```
这个方法使用csv的包读取数据，并将条状的（1\*784）reshape成1\*28\*28的tensor形式，将其和标签合并成一个元组temp放入名为data的list。最后返回预处理好的数据集。
```
 dataset = read_train_data('train.csv')
    train_loader = torch.utils.data.DataLoader(dataset=dataset[0:33600],
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=dataset[33600:],
                                              batch_size=batch_size,
                                              shuffle=True)
```
读取数据集后使用DataLoader方法进行封装数据，利用多进程来加速batch data的处理，使用yield来使用有限的内存。DataLoader是一个高效，简洁，直观的网络输入数据结构，便于使用和扩展。80%的数据用来作为训练集，20%的数据用来作为测试集。
## 训练神经网络模型
```
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
```
class CNN读入nn.Module参数。nn.Module 初始化了一系列重要的成员变量。这些变量初始化了在模块 forward、 backward 和权重加载等时候会被调用的的 hooks，也定义了 parameters 和 buffers。\_\_init\_\_函数继承父类。初始化两层卷积层。  
第一层和第二层卷积层先进行卷积操作，使用5\*5的感受野，移动步长是1，第一层使用16个卷积层，产生16个channel，而第二层使用32个卷积层，产生32个channel。我们希望使用padding为'VALID',根据公式算出padding为2层。接着让产生的数据通过激活函数（ReLU）,再进行归一化处理和dropout处理防止训练模型过拟合，最后在经过一次最大值池化层，减少最后连接层的中的参数数量。  
![image](https://user-images.githubusercontent.com/52622948/159292744-1dadb972-12e4-407f-a9c7-e9090ebda5a2.png)






