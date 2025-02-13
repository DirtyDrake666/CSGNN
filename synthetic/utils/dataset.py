import numpy as np
import igraph as ig 
import networkx as nx
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils.convert import from_networkx, to_networkx
import tqdm
from collections import defaultdict

class ModifData(Data) : 
    def __init__(self, edge_index=None, x=None, *args, **kwargs):
            super().__init__(x=x, edge_index = edge_index, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        
        if 'index' in key or 'face' in key or "path" in key :
            return self.num_nodes
        else:
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key or 'face' in key :#or "path" in key or "indicator" in key:
            return 1
        else:
            return 0

class CSDataset(Dataset) :
    def __init__(self, Gs, features, y, cutoff = 11, min_length = 0, undirected = True, task = 'classification') :
        super().__init__()
        self.Gs = Gs
        self.features = features
        self.y = y 
        self.cutoff = cutoff
        self.undirected = undirected
        self.task = task
        self.diameter = cutoff
        self.min_length = min_length
        self.datalist = [self._create_data(i) for i in range(self.len())]
        
    def len(self) : 
        return len(self.Gs)
    
    def num_nodes(self) : 
        return sum([G.number_of_nodes() for G in self.Gs])

    def _create_data(self, index) : 
        if self.task == 'iso' : 
            x_class = torch.DoubleTensor
        else : 
            x_class = torch.FloatTensor
        data = ModifData(**from_networkx(self.Gs[index]).stores[0])
        data.x = x_class(self.features[index])
        data.y = torch.LongTensor([self.y[index]])


        dis=process_cycles(self.Gs[index],self.cutoff)
        for i in range(self.cutoff):
            if i in dis:
                setattr(data,f'pos_{i}',torch.tensor(dis[i],dtype=torch.long).transpose(0,1))

        dis2=process_cycles(nx.complement(self.Gs[index]),self.cutoff)
        for i in range(self.cutoff):
            if i in dis2:
                setattr(data,f'pos2_{i}',torch.tensor(dis2[i],dtype=torch.long).transpose(0,1))

        nb_list = []
        for edge in self.Gs[index].edges:
            nb_list.append([edge[0], edge[1]])
            nb_list.append([edge[1], edge[0]])
        setattr(data, 'nb_list', torch.tensor(nb_list, dtype=torch.long).transpose(0, 1))
        
        nb2_list=[]
        nb3_list=[]
        for nd in self.Gs[index].nodes:
            #nbs=self.Gs[index].neighbors[nd]
            nbs=findneighbor(nd,nb_list)
            for nb in nbs:
                #nb2s=self.Gs[index].neighbors[nb]
                nb2s=findneighbor(nb,nb_list)
                for nb2 in nb2s:
                    nb2_list.append([nd,nb2])
                    #nb3s=self.Gs[index].neighbors[nb2]
                    nb3s=findneighbor(nb2,nb_list)
                    for nb3 in nb3s:
                    	nb3_list.append([nb,nb3])
        setattr(data, 'nb2_list', torch.tensor(nb2_list, dtype=torch.long).transpose(0, 1))
        setattr(data, 'nb3_list', torch.tensor(nb3_list, dtype=torch.long).transpose(0, 1))
        return data 

    def get(self, index) : 
        return self.datalist[index]
        
def find_shortest_lists(list_of_lists):
    return list_of_lists

def findneighbor(node,list):
    res=[]
    for sublist in list:
        if sublist[0]==node :
            res.append(sublist[1])
        elif sublist[1]==node :
            res.append(sublist[0])
    return res

def process_cycles(graph, cutoff_radius):
    allcycles = nx.chordless_cycles(graph, 2 * cutoff_radius)
    node_to_cycles = {node: [] for node in graph.nodes}

    for cycle in allcycles:
        for nd in cycle:
            node_to_cycles[nd].append(cycle)

    dis = {}
    for nd in node_to_cycles:
        node_to_cycles[nd] = find_shortest_lists(node_to_cycles[nd])
        if node_to_cycles[nd] is None:
            continue
        for cycle in node_to_cycles[nd]:
            startpos = cycle.index(nd)
            l = len(cycle)
            for end in cycle:
                endpos = cycle.index(end)
                distance = min(abs(endpos - startpos), l - abs(endpos - startpos))
                if distance not in dis:
                    dis[distance] = []
                dis[distance].append([nd, end])

    return dis

