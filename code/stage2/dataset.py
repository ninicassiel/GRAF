# -*- coding:UTF-8 -*-
'''
@Date   :{DATE}
'''
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
    def __init__(self, table_path,dmri_graph_dir,dmri_graph_edge_dir, fmri_graph_dir, fmri_feature_dir,
                 data_type='train',fc_threshold=0,sc_threshold=0,fold=5,ROI=100, BOLD=False):
        self.table = table_path
        self.data_type = data_type
        if sc_threshold:
            self.dmri_graph_dir = dmri_graph_dir+ f'_filtered_0{int(sc_threshold*10)}'
            self.dmri_graph_node_dir = dmri_graph_dir+ f'_filtered_0{int(sc_threshold*10)}_nodeFeature_mat_z_scoreNormalized'
        else:
            self.dmri_graph_dir = dmri_graph_dir
            self.dmri_graph_node_dir = dmri_graph_dir+'_nodeFeature_mat'
        self.dmri_graph_edge_dir = dmri_graph_edge_dir
        if fc_threshold:
            self.fmri_graph_dir = fmri_graph_dir + f'_filtered_0{int(fc_threshold*10)}'
        else:
            self.fmri_graph_dir = fmri_graph_dir
        # self.fmri_graph_dir = fmri_graph_dir
        self.fmri_feature_dir = fmri_feature_dir
        self.fc_threshold = fc_threshold
        self.sc_threshold = sc_threshold

        self.table = pd.read_csv(table_path)
        self.table = self.table[self.table['Diag-GUO'] == 'NC']
        if data_type=='train':
            self.table = self.table[self.table['fold'] != fold]
        elif data_type=='val':
            self.table = self.table[self.table['fold'] == fold]
        self.subject_ids = self.table['subject_id'].tolist()
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

    def get_dmri_data(self,dmri_graph_path,dmri_graph_node_path,dmri_graph_edge_path):

        dmri_sc_edge= scipy.io.loadmat(dmri_graph_edge_path)
        dmri_sc_node = scipy.io.loadmat(dmri_graph_node_path)
        d_adj_sc = pd.read_csv(dmri_graph_path, header=None).fillna(0).values

        rows, cols = np.nonzero(d_adj_sc)  # 假设所有图的非零元素位置相同


        d_adj_md = dmri_sc_edge['Md']
        d_adj_rd = dmri_sc_edge['Rd']
        d_adj_ad = dmri_sc_edge['Ad']

        d_adj_md = d_adj_md[rows, cols]
        d_adj_rd = d_adj_rd[rows, cols]
        d_adj_ad = d_adj_ad[rows, cols]
        # 归一化特征
        # d_adj_md = self.normalize_feature(d_adj_md)
        # d_adj_rd = self.normalize_feature(d_adj_rd)
        # d_adj_ad = self.normalize_feature(d_adj_ad)

        edge_index = torch.tensor([rows,cols],dtype=torch.long)
        # 使用 torch.stack 同时处理多个 edge_attr，减少中间变量
        edge_attr = torch.stack([torch.tensor(d_adj_md),
                                 torch.tensor(d_adj_rd),
                                 torch.tensor(d_adj_ad)], dim=1).float()

        DC = dmri_sc_node['DC']
        BC = dmri_sc_node['BC']
        CC = dmri_sc_node['CC']

        node_features = np.concatenate([DC,BC,CC],axis=1)
        node_features = torch.tensor(node_features).float()

        graph = Data(edge_index=edge_index,edge_attr=edge_attr,x=node_features)

        return graph
    def get_fmri_data(self,fmri_graph_path,fmri_feature_path):
        fmri_graph = pd.read_csv(fmri_graph_path, header=None).values
        fmri_feature = pd.read_csv(fmri_feature_path, header=None).values
        # np.fill_diagonal(fmri_graph, 0)
        rows, cols = np.nonzero(fmri_graph)
        # weights = fmri_graph[rows, cols]
        # if self.fc_threshold:
        #     threshold = np.percentile(weights,  self.fc_threshold*100)
        #     filtered_indices = weights >= threshold
        #     filtered_rows = rows[filtered_indices]
        #     filtered_cols = cols[filtered_indices]
        # else:
        #     filtered_rows = rows
        #     filtered_cols = cols
        if self.BOLD==False:
        # # 对数据进行归一化处理
            min_val = fmri_feature.min()
            max_val = fmri_feature.max()
            normalized_data = (fmri_feature - min_val) / (max_val - min_val)
        else:
            normalized_data = fmri_feature
        node_feat = torch.tensor(normalized_data).float()
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        x = node_feat
        edge_attr = torch.tensor(fmri_graph[rows, cols]).float().unsqueeze(-1)
        graph = Data(edge_index=edge_index, x=x,edge_attr=edge_attr)
        return graph


    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        # 读取 dMRI 数据
        dmri_graph_path =f"{self.dmri_graph_dir}/{subject_id}/connectome_strength_pe_s_{self.ROI}_7.csv"
        dmri_graph_edge_path = f"{self.dmri_graph_edge_dir}/{subject_id}/normalized_Ad_Md_Rd_Fa_{self.ROI}.mat"
        dmri_graph_node_path = f"{self.dmri_graph_node_dir}/{subject_id}/normalized_BC_CC_DC_{self.ROI}.mat"

        SC_g = self.get_dmri_data(dmri_graph_path,dmri_graph_node_path,dmri_graph_edge_path)

        # 读取 fMRI 数据
        fmri_graph_path = f"{self.fmri_graph_dir}/{subject_id}/{self.ROI}Parcels_7Networks_ROISignals.csv"
        fmri_feature_path = f"{self.fmri_feature_dir}/{subject_id}.csv"
        FC_g = self.get_fmri_data(fmri_graph_path, fmri_feature_path)
        return {
                'subject_id': subject_id,
                "SC_graph":SC_g,
                "FC_graph":FC_g
        }

class MyGraphDataset_withPET(MyGraphDataset):
    def __init__(self,table_path,dmri_dir, fmri_graph_dir, fmri_feature_dir,
                 PET_dir,data_type='train',fold=5,ROI=100,BOLD=False):
        super(MyGraphDataset_withPET, self).__init__(table_path,dmri_dir, fmri_graph_dir, fmri_feature_dir,data_type=data_type,fold=fold,ROI=ROI,BOLD=BOLD)
        self.PET_dir = PET_dir
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
        label_path = os.path.join(self.PET_dir, subject_id, f'{self.ROI}Parcels_7Networks_ROISignals_{subject_id}.mat')
        pet_label = self.get_PET_data(label_path)

        return {
            'subject_id': subject_id,
            'SC_graph': SC_graph,
            'FC_graph': FC_graph,
            'PET_label': pet_label
        }

def mask_edge(edge_index,mask_ratio=0.7,seed=0):

    # 获取单向边
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
    unmasked_edges = torch.tensor(list(remain_edge_set)).t().contiguous()
    # 保证无向性
    return masked_edges,unmasked_edges

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
        masked_edge_index = to_undirected(masked_edge_index)

    elif num_mask_edge < neg_num:
        align_num = num_mask_edge
        align_index = torch.randperm(neg_num)[:align_num]
        neg_edge_index = neg_edge_index[:, align_index]
        neg_edge_index = to_undirected(neg_edge_index)

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
    mask = torch.full((edge_index.size(1),), float('nan'))
    mask[mask_index] = 1
    mask[remain_index] = 0
    graph.edge_mask = mask

    new_edge_attr = torch.full((edge_index.size(1), edge_attr.size(1)), float('nan'), device=edge_attr.device)

    # 将旧的 edge_attr 复制到新矩阵的前部分
    new_edge_attr[:edge_attr.size(0)] = edge_attr
    graph.edge_attr = new_edge_attr
    return graph
def collate_fn(batch_data,mask_ratio=None,fc=True,sc=True,seed =0):
    subject_ids = [i['subject_id'] for i in batch_data]
    FC_graphs, SC_graphs = None, None
    if fc:
        FC_graphs = []
        for i in batch_data:
            graph = i['FC_graph']
            if mask_ratio:
                graph = Get_mask_neg(graph,mask_ratio=mask_ratio,seed = seed)
            FC_graphs.append(graph)
        FC_graphs = Batch.from_data_list(FC_graphs)
    if sc:
        SC_graphs = []
        for i in batch_data:
            graph = i['SC_graph']
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

if __name__ == '__main__':
    from functools import partial
    from torch.utils.data import DataLoader
    import numpy as np

    # table_path = "D:/yhy/data/huashan/T#019fmriROISignalRecoder_woHeadPoor_100_withAbeta_dmri_resplit_fold.csv"
    # dmri_graph_dir = 'D:/yhy/data/huashan/014_dmri_sc'
    # dmri_graph_edge_dir = 'D:/yhy/data/huashan/010_dmri_sc_AdMdRdFa_mat_z_scoreNormalized'
    # fmri_graph_dir = 'D:/yhy/data/huashan/011_fmri_FC_bsNormalizedSignal'
    # fmri_feature_dir = 'D:/yhy/data/huashan/004_fmriSignalCSV_feature_CNNAE/AEfeature_ROI100'
    # PET_dir = 'D:/yhy/data/huashan/013_PET_suvr'

    table_path = "/public_bme/data/yuanhy/d#001_Pet_center/Huashan/T#019fmriROISignalRecoder_woHeadPoor_100_withAbeta_dmri_resplit_fold.csv"
    dmri_graph_dir = '/public_bme/data/yuanhy/d#001_Pet_center/Huashan/014_dmri_sc'
    dmri_graph_edge_dir = '/public_bme/data/yuanhy/d#001_Pet_center/Huashan/010_dmri_sc_AdMdRdFa_mat_z_scoreNormalized'
    fmri_graph_dir = '/public_bme/data/yuanhy/d#001_Pet_center/Huashan/011_fmri_FC_bsNormalizedSignal'
    fmri_feature_dir = '/public_bme/data/yuanhy/d#001_Pet_center/Huashan/004_fmriSignalCSV_feature_CNNAE/AEfeature_ROI100'
    PET_dir = '/public_bme/data/yuanhy/d#001_Pet_center/Huashan/013_PET_suvr'

    fold = 1
    ROI = 100
    BOLD = False

    sc_threshold = 0.4
    fc_threshold = 0.4
    mask_ratio = 0.2
    # for sc_threshold in np.arange(0.1, 0.6, 0.1):
    #     fc_threshold = sc_threshold
    #     for mask_ratio in np.arange(0.1, 0.6, 0.1):

    val_dataset = MyGraphDataset(table_path, dmri_graph_dir, dmri_graph_edge_dir,
                                 fmri_graph_dir, fmri_feature_dir,
                                 fc_threshold=fc_threshold, sc_threshold=sc_threshold,
                                 data_type='val', fold=fold, ROI=ROI, BOLD=BOLD)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            collate_fn=partial(collate_fn,
                                               mask_ratio=mask_ratio,
                                               fc=True,
                                               sc=True,
                                               seed=0))
    import time
    start = time.time()
    for epoch in range(20):
        if epoch % 5 == 0:
            seed = epoch // 5
            random.seed(seed)
            torch.manual_seed(seed)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                    collate_fn=partial(collate_fn,
                                                       mask_ratio=mask_ratio,
                                                       fc=True,
                                                       sc=True,
                                                       seed=seed))
        for batch_data in val_loader:
            sids = batch_data['subject_ids']
            fc_graphs = batch_data['FC_graphs']
            sc_graphs = batch_data['SC_graphs']
            fc_graphs_list = Batch.to_data_list(fc_graphs)
            sc_graphs_list = Batch.to_data_list(sc_graphs)
            print(len(fc_graphs_list))
            for index,graph in enumerate(sc_graphs_list):
                mask_edge_index = graph.edge_index[:,graph.edge_mask==1]
                remain_edge_index = graph.edge_index[:,graph.edge_mask==0]
                neg_edge_index = graph.edge_index[:,graph.neg_mask==1]
                print('mask edge',mask_edge_index.size(1))
                print('remain edge',remain_edge_index.size(1))
                print('neg edge',neg_edge_index.size(1))
                from tools.visualization_graphs import visualiaze_mask_hitmap
                save_path = os.path.join('D:/Projects/p#001DTI2PET/dmri_fmri2PET/mymodels/PyG_models_v8_2/visualize',f'{sids[index]}_mask_hitmap_sc_threshold{sc_threshold:.1}_maskRatio{mask_ratio:.1}_epoch{epoch}.png')
                os.makedirs(os.path.dirname(save_path),exist_ok=True)
                print(f'{sids[index]}_mask_hitmap_sc_threshold{sc_threshold:.1}_maskRatio{mask_ratio:.1}')
                visualiaze_mask_hitmap(pos_edge_index=mask_edge_index,
                                       neg_edge_index=neg_edge_index,
                                       remain_edge_index=remain_edge_index,
                                       num_node=graph.num_nodes,
                                       save_path=save_path,
                                       show=False)

                print('end')
                break
            for index,graph in enumerate(fc_graphs_list):
                mask_edge_index = graph.edge_index[:, graph.edge_mask == 1]
                remain_edge_index = graph.edge_index[:, graph.edge_mask == 0]
                neg_edge_index = graph.edge_index[:, graph.neg_mask == 1]
                print('mask edge', mask_edge_index.size(1))
                print('remain edge', remain_edge_index.size(1))
                print('neg edge', neg_edge_index.size(1))
                from tools.visualization_graphs import visualiaze_mask_hitmap

                save_path = os.path.join('D:/Projects/p#001DTI2PET/dmri_fmri2PET/mymodels/PyG_models_v8_2/visualize',
                                         f'{sids[index]}_mask_hitmap_fc_threshold{sc_threshold:.1}_maskRatio{mask_ratio:.1}_epoch{epoch}.png')
                print(f'{sids[index]}_mask_hitmap_fc_threshold{sc_threshold:.1}_maskRatio{mask_ratio:.1}')
                visualiaze_mask_hitmap(pos_edge_index=mask_edge_index,
                                       neg_edge_index=neg_edge_index,
                                       remain_edge_index=remain_edge_index,
                                       num_node=graph.num_nodes,
                                       save_path=save_path,
                                       show=False)
                print('end')
                break
            break
        print('epoch:',epoch)
    end = time.time()
    print('time:',end-start)





