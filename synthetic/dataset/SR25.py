import networkx as nx
import torch
from utils.dataset import CSDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
import tqdm 
import igraph as ig 
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

class SR25(torch.utils.data.Dataset):
    
    def __init__(self,device, root="../data/SR25/",dataset_name = "sr16622.g6", task='iso',cutoff = 5):
        self.name = "SR25"
        self.device=device
        self.cutoff = cutoff
        self.root = root + dataset_name
        self.task = task
        self._prepare()
    def _prepare(self):

        Gs = nx.read_graph6(self.root)
        ys = [i for i in range(Gs.__len__())]
        #print(ys)
        features = []
        for G in Gs:
            features.append(np.ones((nx.number_of_nodes(G), 1)))
            
        self.dataset = CSDataset(Gs, features, ys ,self.cutoff, task=self.task)
        

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__" : 
    dataset = SR25()
    Gs=nx.read_graph6('../data/SR25/sr16622.g6')
    fig, axs = plt.subplots(1, len(Gs), figsize=(15, 5))

    # 遍历每个图并在子图中绘制
    for i, graph in enumerate(Gs):
        pos = nx.spring_layout(graph)  # 确定节点的位置
        nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=500, font_size=10, ax=axs[i])
        axs[i].set_title(f'Graph {i + 1}')  # 设置子图标题

    plt.tight_layout()
    plt.show()



        
