import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#  训练经验 少量多次，层数少
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
BATCH_SIZE = 5
EPOCHS = 1000
bidirectional_set = True
bidirectional = 2 if bidirectional_set else 1
input_dim = 23
hidden_dim1 = 15
hidden_dim2 = 5
layer_dim = 1
output_dim = 2
sequence_dim = 1
learning_rate = 0.001
loss_list = []
accracy_list = []
iteration_list = []


class CSVDataSet(Dataset):
    def __init__(self, filrpath):
        print(f'reading{filrpath}')

        df = pd.read_csv(
            filrpath, header=0, index_col=0,
            encoding='utf-8',
            dtype={'label': np.int32}
        )

        feat = df.iloc[:, 2:].values
        print(f'the shape of feature is {feat.shape}')
        label = df.iloc[:, 0].values


        self.x = torch.from_numpy(feat).to(torch.float32)
        self.y = torch.from_numpy(label)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class LSTM_Model(nn.Module):
    def __init__(self, my_input_dim, my_hidden_dim1, my_hidden_dim2, my_layer_dim, my_output_dim):
        super(LSTM_Model, self).__init__()
        self.hidden_dim1 = my_hidden_dim1
        self.hidden_dim2 = my_hidden_dim2
        self.layer_dim = my_layer_dim
        # 构建lstm模型
        self.lstm = nn.LSTM(my_input_dim, my_hidden_dim1, my_layer_dim, batch_first=True,
                            bidirectional=bidirectional_set)
        self.dropout = nn.Dropout(p=0.5)  # dropout训练
        self.lstm2 = nn.LSTM(my_hidden_dim1*bidirectional, my_hidden_dim2, my_layer_dim, batch_first=True,
                            bidirectional=bidirectional_set)
        # 全连接层
        self.fc = nn.Linear(my_hidden_dim2 * bidirectional, my_output_dim)

    def forward(self, x):
        # 初始化隐层状态全为0
        h1 = torch.zeros(self.layer_dim * bidirectional, x.size(0), self.hidden_dim1).requires_grad_().to(device)
        c1 = torch.zeros(self.layer_dim * bidirectional, x.size(0), self.hidden_dim1).requires_grad_().to(device)
        # 分离隐藏状态，避免梯度爆炸
        out, (hn, cn) = self.lstm(x, (h1.detach(), c1.detach()))  # 将输入数据和初始化隐层、记忆单元信息传入

        out = self.dropout(out)

        h2 = torch.zeros(self.layer_dim * bidirectional, out.size(0), self.hidden_dim2).requires_grad_().to(device)
        c2 = torch.zeros(self.layer_dim * bidirectional, out.size(0), self.hidden_dim2).requires_grad_().to(device)
        # 分离隐藏状态，避免梯度爆炸
        out_lstm2, (hn, cn) = self.lstm2(out, (h2.detach(), c2.detach()))  # 将输入数据和初始化隐层、记忆单元信息传入

        # 以最后一层隐层状态为输出
        out_liner = self.fc(out_lstm2[:, -1, :])

        return out_liner, out_lstm2


if __name__ == "__main__":
    train_sets = CSVDataSet(r'E:\城市与区域生态\大熊猫和竹\种群动态模拟\竹分布\三调bamboo.csv ')  # E:\城市与区域生态\大熊猫和竹\种群动态模拟\竹分布\三调bamboo.csv
    test_sets = CSVDataSet(r'E:\城市与区域生态\大熊猫和竹\种群动态模拟\竹分布\三调bamboo_test.csv')
    train_loader = DataLoader(train_sets, BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_sets, BATCH_SIZE, shuffle=True)

    model = LSTM_Model(input_dim, hidden_dim1, hidden_dim2, layer_dim, output_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    iter = 0
    for epoch in range(EPOCHS):
        for i, (wdrvi, labels) in enumerate(train_loader):

            model.train()
            wdrvi = wdrvi.view(-1, sequence_dim, input_dim).requires_grad_().to(device)

            labels = labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            #  前向传播
            outputs, _ = model(wdrvi)
            #  计算损失
            loss = criterion(outputs, labels)
            #  反向传播
            loss.backward()
            #  更新参数
            optimizer.step()
            iter += 1
            #  模型验证
            if iter % 500 == 0:
                model.eval()
                correct = 0
                total = 0
                for wdrvi, labels in test_loader:
                    wdrvi = wdrvi.view(-1, sequence_dim, input_dim).requires_grad_().to(device)
                    outputs, out_of_lstm = model(wdrvi)
                    # print(outputs)
                    predict = torch.max(outputs, 1)[1]
                    # print(predict)
                    total += labels.size(0)
                    correct += (predict == labels.to(device)).sum()

                accrucy = correct / total
                loss_list.append(loss.data.cpu())
                accracy_list.append(accrucy.cpu())
                iteration_list.append(iter)

                print('loop:{} Loss:{} Accuracy:{}'.format(iter, loss.item(), accrucy))
    plt.plot(iteration_list, loss_list)
    plt.xlabel('Number of Iteraion')
    plt.ylabel('Loss')
    plt.show()

    plt.plot(iteration_list, accracy_list)
    plt.xlabel('Number of Iteraion')
    plt.ylabel('Accuracy')
    plt.show()

    PATH = r'E:\城市与区域生态\大熊猫和竹\种群动态模拟\竹分布\三调model_b5_epo1k_h1_23_h25_lr1e-3.pt'
    torch.save(model,PATH)
