import networkx as nx
import torch
import torch.nn as nn
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.utils import to_networkx
'''
class CSGS(GlobalStorage):
    def __init__(self,original_obj,pos1=None,pos2=None,pos3=None):
        super().__init__()
        self.__dict__.update(original_obj.__dict__)
        self.pos1=pos1
        self.pos2=pos2
        self.pos3=pos3
'''

def _process(dataset):
    new_dataset = []
    for i in range(len(dataset)):
        data = dataset[i]
        G = to_networkx(data, to_undirected=True)
        node_cycle_counts = torch.zeros((data.num_nodes, 8), dtype=torch.long)
        for cycle in nx.chordless_cycles(G,11):
            cycle_length = len(cycle)
            if cycle_length in [4, 5, 6,7,8,9,10,11]:
                for node in cycle:
                    if node_cycle_counts[node, cycle_length - 4] ==0:
                        node_cycle_counts[node, cycle_length - 4] += 1

        data.node_cycle_counts = node_cycle_counts
        new_dataset.append(data)

    return new_dataset

class Logger(object):
    def __init__(self, file,log_file):
        self.terminal = file
        self.log = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass
class Trainer:
    def __init__(self, model, train_loader, test_loader, device, lr=0.001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = torch.nn.L1Loss()

    def train(self):
        self.model.train()
        total_loss = 0
        for data in self.train_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            out = self.model(data).squeeze()
            loss = self.loss_fn(out, data.y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.train_loader)

    def test(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in self.test_loader:
                data = data.to(self.device)
                out = self.model(data).squeeze()
                loss = self.loss_fn(out, data.y)
                total_loss += loss.item()
        return total_loss / len(self.test_loader)