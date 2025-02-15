import torch
import torch.nn.functional as F
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GENConv, global_add_pool
from preprocess import _process, Logger, Trainer
import os
import sys
from datetime import datetime

# 定义一个简单的GEN模型
torch.manual_seed(2025)
class GEN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5, lstm_hidden_size=64, lstm_num_layers=1):
        super(GEN, self).__init__()
        self.fc = torch.nn.Linear(in_channels + 8, hidden_channels)
        self.edge_emb = torch.nn.Embedding(4, hidden_channels)
        self.convs = torch.nn.ModuleList([GENConv(hidden_channels, hidden_channels) for _ in range(num_layers)])
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)])
        self.lstm = torch.nn.LSTM(hidden_channels, lstm_hidden_size, lstm_num_layers, batch_first=True)
        self.lin = torch.nn.Linear(lstm_hidden_size, out_channels)
        self.num_layers = num_layers
        


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr =self.edge_emb(data.edge_attr)
        x = torch.cat((x, data.node_cycle_counts), dim=1)
        x = x.float()
        x = self.fc(x)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index,edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)

        x = global_add_pool(x, batch)
        x = x.unsqueeze(1)  # 添加时间步维度
        x, (h_n, c_n) = self.lstm(x)
        x = x.squeeze(1)  # 移除时间步维度
        x = self.lin(x)
        return x

# 加载 ZINC 数据集
dataset = ZINC(root='data/ZINC', subset=False)
train_dataset = ZINC(root='data/ZINC', split='train', subset=False)  # 训练集
test_dataset = ZINC(root='data/ZINC', split='test', subset=False)  # 测试集
train_dataset, test_dataset = _process(train_dataset), _process(test_dataset)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GEN(in_channels=dataset.num_features, hidden_channels=128, out_channels=1).to(device)
lr = 0.001
trainer = Trainer(model, train_loader, test_loader, device, lr=lr)

if not os.path.exists("log"):
    os.makedirs("log")
# 打开 log 文件
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = f"log/GEN+cycle_lr_{lr}_{timestamp}.txt"
log_file = open(log_file_path, "w")
sys.stdout = Logger(sys.stdout, log_file)
best_loss = 1
# 训练和测试模型
for epoch in range(1, 1001):
    train_loss = trainer.train()
    test_loss = trainer.test()
    best_loss = min(test_loss, best_loss)
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test MAE: {test_loss:.4f}, Best MAE: {best_loss:.4f}')
