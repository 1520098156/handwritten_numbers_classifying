import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cnn  # 自己建立的cnn

# 定义超参数
input_size = 28  # 图像的总尺寸28*28
num_classes = 10  # 标签的种类数
num_epochs = 3  # 训练的总循环周期
batch_size = 64  # 一个撮（批次）的大小，64张图片


def accuracy(predictions, labels):
    pred = torch.max(predictions.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


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


if __name__ == '__main__':
    dataset = read_train_data('train.csv')
    train_loader = torch.utils.data.DataLoader(dataset=dataset[0:33600],
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=dataset[33600:],
                                              batch_size=batch_size,
                                              shuffle=True)

    # 实例化
    net = cnn.CNN()
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器，普通的随机梯度下降算法

    right_rate = []
    right_num = 0

    for epoch in range(num_epochs):
        print('epoch=', epoch)
        # 当前epoch的结果保存下来
        train_rights = []

        for batch_index, (data, target) in enumerate(train_loader):
            net.train()
            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            right = accuracy(output, target)
            train_rights.append(right)

            if batch_index % 100 == 0:

                net.eval()
                val_rights = []

                for (data, target) in test_loader:
                    output = net(data)
                    right = accuracy(output, target)
                    val_rights.append(right)

                # 准确率计算
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))

                print('当前epoch: {} [{}/{} ({:.0f}%)]\t损失: {:.6f}\t训练集准确率: {:.2f}%\t测试集正确率: {:.2f}%'.format(
                    epoch, batch_index * batch_size, len(train_loader.dataset),
                           100. * batch_index / len(train_loader),
                    loss.data,
                           100. * train_r[0].numpy() / train_r[1],
                           100. * val_r[0].numpy() / val_r[1]))
