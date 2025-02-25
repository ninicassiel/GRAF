# -*- coding:UTF-8 -*-
'''
@Date   :{DATE}
'''
import time

import scipy.io
from torch_geometric.data import Batch
import warnings
import torch
from torch.utils.data import Dataset
import random
import numpy as np
import pandas as pd
import os
import scipy.io as sio
from typing import Optional
from torch_geometric.data import Data
import torch_cluster  # noqa
random_walk = torch.ops.torch_cluster.random_walk
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import to_undirected,sort_edge_index,degree
warnings.filterwarnings("ignore")

class MyGraphDataset(Dataset):
    def __init__(self, table_path,
                 dmri_graph_dir,
                 fmri_graph_dir,
                 data_type='train',fc_threshold=0,sc_threshold=0,fold=5,ROI=100, BOLD=False):
        self.table = table_path
        self.data_type = data_type
        if sc_threshold:
            self.dmri_graph_dir = dmri_graph_dir+ f'_filtered_0{int(sc_threshold*10)}_ROI{ROI}_pt'
        else:
            self.dmri_graph_dir = dmri_graph_dir
        if fc_threshold:
            self.fmri_graph_dir = fmri_graph_dir + f'_AE_filtered_0{int(fc_threshold*10)}_ROI{ROI}_pt'
        else:
            self.fmri_graph_dir = fmri_graph_dir
        self.fc_threshold = fc_threshold
        self.sc_threshold = sc_threshold

        self.table = pd.read_csv(table_path)
        self.table = self.table[self.table['Diag-GUO'] == 'NC']
        if data_type=='train':
            self.table = self.table[self.table['fold'] != fold]
        elif data_type=='val':
            self.table = self.table[self.table['fold'] == fold]
        self.subject_ids = self.table['subject_id'].tolist()
        remove_list = [
            'AFM0535'
        ]
        if self.data_type == 'val':
            for i in remove_list:
                if i in self.subject_ids:
                    self.subject_ids.remove(i)
        self.BOLD = BOLD
        self.ROI =ROI
    def __len__(self):
        return len(self.subject_ids)

    def normalize_feature(self,data):
        mask = (data != 0).astype(np.float32)
        min_val = data.min()
        max_val = data.max()
        normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data * mask

    def get_dmri_data(self,dmri_graph_path):
        graph = torch.load(dmri_graph_path)
        edge_index = graph['edge_index']
        edge_attr = graph['edge_features']
        node_features = graph['node_features']
        graph = Data(edge_index=edge_index,edge_attr=edge_attr,x=node_features)

        return graph
    def get_fmri_data(self,fmri_graph_path,):
        graph = torch.load(fmri_graph_path)
        edge_index = graph['edge_index']
        edge_attr = graph['edge_features']
        node_features = graph['node_features']
        graph = Data(edge_index=edge_index, x=node_features,edge_attr=edge_attr)

        return graph


    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        # 读取 dMRI 数据
        dmri_graph_path =f"{self.dmri_graph_dir}/{subject_id}.pt"
        SC_g = self.get_dmri_data(dmri_graph_path)

        # 读取 fMRI 数据
        fmri_graph_path = f"{self.fmri_graph_dir}/{subject_id}.pt"
        FC_g = self.get_fmri_data(fmri_graph_path)
        return {
                'subject_id': subject_id,
                "SC_graph":SC_g,
                "FC_graph":FC_g
        }

class MyGraphDataset_withPET(MyGraphDataset):
    def __init__(self, table_path,dmri_graph_dir, fmri_graph_dir,PET_dir,
                 data_type='train',fc_threshold=0,sc_threshold=0,fold=5,ROI=100, BOLD=False,pick=False):
        super(MyGraphDataset_withPET, self).__init__(table_path=table_path,
                                                     dmri_graph_dir = dmri_graph_dir,
                                                     fmri_graph_dir = fmri_graph_dir,
                                                     data_type=data_type,
                                                     fc_threshold=fc_threshold,
                                                     sc_threshold=sc_threshold,
                                                     fold=fold,ROI=ROI, BOLD=BOLD)
        self.PET_dir = PET_dir
        self.table = pd.read_csv(table_path)
        # self.table = pd.read_csv(table_path)
        if data_type == 'train':
            self.table = self.table[self.table['fold'] != fold]
        elif data_type == 'val':
            self.table = self.table[self.table['fold'] == fold]
        self.subject_ids = self.table['subject_id'].tolist()
        remove_list = [
            'AFM0535'
        ]
        if self.data_type == 'val':
            for i in remove_list:
                if i in self.subject_ids:
                    self.subject_ids.remove(i)
        if pick:
            remove_list = ['AFM0169', 'AFM0006', 'AFM0143', 'AFM0052', 'AFM0347', 'AFM0175', 'AFM0333', 'AFM0478',
                           'AFM0561', 'AFM0132',
                           'AFM0395', 'AFM0245', 'AFM0099', 'AFM0531',
                           'AFM0367', 'AFM0236', 'AFM0178', 'AFM0557',
                           'AFM0076', 'AFM0044', 'AFM0538', 'AFM0576',
                           'AFM0535', 'AFM0176', 'AFM0502', 'AFM0665',
                           ]
            if self.data_type == 'val':
                for i in remove_list:
                    if i in self.subject_ids:
                        self.subject_ids.remove(i)
    def get_PET_data(self,PET_data_path):

        # 加载 PET SUVR 数据
        label_data = sio.loadmat(PET_data_path)['ROISignals']  # 假设变量名为 'ROISignals'

        # 将 label 数据转换为 PyTorch Tensor
        label = torch.tensor(label_data, dtype=torch.float32).squeeze()
        return label
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        subject_id = data['subject_id']
        SC_graph = data['SC_graph']
        FC_graph = data['FC_graph']
        label_path = os.path.join(self.PET_dir, subject_id+'.pt')
        pet_label = torch.load(label_path)

        return {
            'subject_id': subject_id,
            'SC_graph': SC_graph,
            'FC_graph': FC_graph,
            'PET_label': pet_label
        }

def mask_edge(edge_index,mask_ratio=0.7,seed=0):
    row, col = edge_index
    mask = row < col  # 仅保留一组无向边
    undirected_edge_index = edge_index[:, mask]
    # 设置随机数种子，确保可重复性
    torch.manual_seed(seed)

    # 随机选择哪些边要被 mask
    e_ids = torch.arange(undirected_edge_index.size(1), dtype=torch.long, device=edge_index.device)
    mask_prob = torch.full_like(e_ids, mask_ratio, dtype=torch.float32)
    mask = torch.bernoulli(mask_prob).to(torch.bool)

    # 获取 mask 和 remaining 边
    masked_edges = undirected_edge_index[:, mask]
    masked_edges = to_undirected(masked_edges)

    all_edges_set = set(map(tuple, edge_index.T.tolist()))
    masked_edges_set = set(map(tuple, masked_edges.T.tolist()))
    remain_edge_set = all_edges_set - masked_edges_set
    unmasked_edges = torch.tensor(list(remain_edge_set),device=edge_index.device).t().contiguous()
    # 保证无向性
    return masked_edges, unmasked_edges

def generate_full_neg_edges(num_nodes, edge_index,num_neg_samples=None,seed=0):

    row, col = torch.triu_indices(num_nodes, num_nodes, offset=1)
    all_possible_edges = set(zip(row.tolist(), col.tolist()))
    all_possible_edges = {edge for edge in all_possible_edges if edge[0] != edge[1]}

    # Convert edge_index to set for faster membership checking
    edge_set = set(map(tuple, edge_index.t().tolist()))

    # Find non-existing edges
    non_edge_pairs = list(all_possible_edges - edge_set)

    # Convert to tensor
    non_edge_pairs = torch.tensor(non_edge_pairs, dtype=torch.long).t()

    # Sample negative edges if num_neg_samples is specified
    if num_neg_samples:
        num_edges = num_neg_samples // 2
    else:
        num_edges = edge_index.size(1) // 2
    # 设置随机数种子，确保可重复性
    torch.manual_seed(seed)
    # Randomly sample edges
    random_indices = torch.randperm(non_edge_pairs.size(1))[:num_edges]
    neg_edge_pairs = non_edge_pairs[:, random_indices]

    # Ensure undirected edges
    neg_edge_pairs = to_undirected(neg_edge_pairs, num_nodes=num_nodes).to(edge_index.device)

    return neg_edge_pairs


def Get_mask_neg(graph,mask_ratio=0.7,seed = 0):
    device = graph.edge_index.device
    torch.manual_seed(0)
    edge_index = graph.edge_index
    edge_attr  = graph.edge_attr
    masked_edge_index, remain_edge_index = mask_edge(edge_index, mask_ratio=mask_ratio,seed=seed)
    num_nodes = graph.num_nodes
    num_mask_edge = masked_edge_index.size(1)
    neg_edge_index = generate_full_neg_edges(num_nodes, edge_index,
                                             num_neg_samples=num_mask_edge,seed=seed)
    # 确保正负样本数量相等
    neg_num = neg_edge_index.size(1)
    if num_mask_edge > neg_num:
        align_num = neg_num
        # 设置随机数种子，确保可重复性
        noise = torch.randperm(num_mask_edge)
        align_index = noise[:align_num]
        t_masked_edge_index = masked_edge_index[:, align_index]
        masked_edge_index = t_masked_edge_index
        # masked_edge_index = to_undirected(masked_edge_index)
        masked_edge_index = to_undirected(masked_edge_index).to(device)

    elif num_mask_edge < neg_num:
        align_num = num_mask_edge
        align_index = torch.randperm(neg_num)[:align_num]
        neg_edge_index = neg_edge_index[:, align_index]
        # neg_edge_index = to_undirected(neg_edge_index)
        neg_edge_index = to_undirected(neg_edge_index).to(device)

    # 更新edge：将负样本加入到edge_index中
    edge_index = torch.cat((edge_index, neg_edge_index), dim=1)
    graph.edge_index = edge_index
    # 生成neg mask
    neg_mask = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    neg_mask[-neg_edge_index.size(1):] = 1
    graph.neg_mask = neg_mask
    # 生成pos mask
    edge_index_pairs = edge_index.t().unsqueeze(0)
    masked_edge_pairs = masked_edge_index.t().unsqueeze(1)
    remain_edge_pairs = remain_edge_index.t().unsqueeze(1)
    mask_index = (masked_edge_pairs == edge_index_pairs).all(dim=-1).any(dim=0)
    remain_index = (remain_edge_pairs == edge_index_pairs).all(dim=-1).any(dim=0)
    # mask = torch.full((edge_index.size(1),), float('nan'))
    mask = torch.full((edge_index.size(1),), float('nan'), device=device)

    mask[mask_index] = 1
    mask[remain_index] = 0
    graph.edge_mask = mask

    new_edge_attr = torch.full((edge_index.size(1), edge_attr.size(1)), float('nan'), device=edge_attr.device)

    # 将旧的 edge_attr 复制到新矩阵的前部分
    new_edge_attr[:edge_attr.size(0)] = edge_attr
    graph.edge_attr = new_edge_attr
    return graph

def collate_fn(batch_data,mask_ratio=None,fc=True,sc=True,seed =0,device ='cpu'):

    subject_ids = [i['subject_id'] for i in batch_data]
    FC_graphs, SC_graphs = None, None
    if fc:
        FC_graphs = []
        for i in batch_data:
            graph = i['FC_graph']
            graph = graph.to(device)
            if mask_ratio:
                graph = Get_mask_neg(graph,mask_ratio=mask_ratio,seed = seed)
            FC_graphs.append(graph)
        FC_graphs = Batch.from_data_list(FC_graphs)
    if sc:
        SC_graphs = []
        for i in batch_data:
            graph = i['SC_graph']
            graph = graph.to(device)
            if mask_ratio:
                graph = Get_mask_neg(graph,mask_ratio=mask_ratio,seed = seed)
            SC_graphs.append(graph)
        SC_graphs = Batch.from_data_list(SC_graphs)

    # PET labels (if present)
    if 'PET_label' in batch_data[0].keys():
        PET_labels = torch.stack([i['PET_label'] for i in batch_data], dim=0)
        return {
            'subject_ids': subject_ids,
            'SC_graphs': SC_graphs,
            'FC_graphs': FC_graphs,
            'PET_labels': PET_labels
        }

    # Return without PET labels
    return {
        'subject_ids': subject_ids,
        'SC_graphs': SC_graphs,
        'FC_graphs': FC_graphs
    }






