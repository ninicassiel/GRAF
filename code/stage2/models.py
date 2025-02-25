from typing import Union

import torch
from torch import Tensor
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_sparse import SparseTensor
from torch_scatter import scatter


class NodeEdgeInteraction_node(MessagePassing):
    def __init__(self, in_node_channels, in_edge_channels, out_node_channels, out_edge_channels,p=0.5):
        super().__init__(aggr='sum')
        # Linear transformations for node and edge features
        self.node_lin = Linear(in_node_channels, out_node_channels // 2, bias=False)
        self.edge_lin = Linear(in_edge_channels, out_edge_channels // 2, bias=False)
        # # 用于计算edge to node weighted average的权重
        self.edge_weight_lin = Linear(out_edge_channels // 2, 1)
        self.node_bias = Parameter(torch.empty(out_node_channels // 2))
        self.edge_bias = Parameter(torch.empty(out_edge_channels // 2))
        self.final_node_lin = Linear(out_node_channels, out_node_channels)  # 用于对聚合后的节点特征进行处理
        self.final_edge_lin = Linear(out_edge_channels, out_edge_channels)  # 用于对聚合后的边特征进行处理
        self.dropout = torch.nn.Dropout(p=p)
        self.reset_parameters()

    def reset_parameters(self):
        self.node_lin.reset_parameters()
        self.edge_lin.reset_parameters()
        self.final_node_lin.reset_parameters()  # 重置最终的线性变换参数
        self.final_edge_lin.reset_parameters()  # 重置最终的线性变换参数
        self.node_bias.data.zero_()
        self.edge_bias.data.zero_()

    def forward(self, x, edge_index,edge_attr):
        # Step 1: Linear transformations
        node_embedding1 = F.relu(self.node_lin(x))
        if self.training:
            node_embedding1 = self.dropout(node_embedding1)
        edge_embedding1 = F.relu(self.edge_lin(edge_attr))
        if self.training:
            edge_embedding1 = self.dropout(edge_embedding1)

        edge_embedding2 = self.edge_updater(edge_index, edge_embedding1, node_embedding1)

        node_embedding2 = self.propagate(edge_index,
                                         x=node_embedding1,
                                         edge_attr=edge_embedding1)
        # node_embedding3 = self.self_prop(edge_index,
        #                                  x=node_embedding1,
        #                                  edge_attr=edge_embedding1)
        node_embedding2 = F.relu(self.final_node_lin(node_embedding2))  # 线性变换和激活函数
        edge_embedding2 = F.relu(self.final_edge_lin(edge_embedding2))  # 线性变换和激活函数
        return node_embedding2,edge_embedding2
    # def self_prop(self,edge_index,x,edge_attr):
    #     aggr_to_src = scatter(edge_attr, edge_index[0], dim=0, dim_size=x.size(0), reduce=self.aggr)
    #     return torch.cat([aggr_to_src, x], dim=-1)

    def message(self, edge_attr):
        '''
        自定义每条边的信息
        :param edge_attr:
        :return:
        '''
        return edge_attr
    # def aggregate(self, msg, edge_index,num_nodes):
    #     '''
    #     如果定义了message，没必要定义本aggregate函数，因为默认的aggregate函数是将message聚合到目标节点
    #     并且我的这里的是无向图，聚合到目标节点和源节点是一样的
    #     :param msg:
    #     :param edge_index:
    #     :param num_nodes:
    #     :return:
    #     '''
    #     return scatter(msg, edge_index[0], dim=0, dim_size=num_nodes, reduce=self.aggr)
    def update(self, aggr_out,x,edge_index):
        # 如果不定义message 和 aggregate函数，那么
        # msg = x[edge_index[1]] # 每条边的信息来自于目标节点
        # aggr_out = scatter(msg, edge_index[0], dim=0, dim_size=x.size(0), reduce=self.aggr) # 从边聚合到节点的信息是源节点的信息
        return torch.cat([aggr_out, x], dim=-1)

    def edge_updater(self, edge_index, edge_attr, x):
        # 利用节点到边的信息来更新边的信息
        return torch.cat([x[edge_index[0]]+ x[edge_index[1]], edge_attr], dim=-1)
class DotEdgeDecoder(nn.Module):
    """Simple Dot Product Edge Decoder"""

    def __init__(self, *args, **kwargs):
        super().__init__()

    def reset_parameters(self):
        return

    def forward(self, z, edge, sigmoid=True):
        x = z[edge[0]] * z[edge[1]]
        x = x.sum(-1)

        if sigmoid:
            return x.sigmoid()
        else:
            return x

class GNNEncoder(nn.Module):
    def __init__(self, in_node_channels,
                 in_edge_channels,
                 hidden_channels,
                 out_node_channels,
                 out_edge_channels,
                 num_layers,
                 dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = num_layers
        self.dropout = dropout
        # First layer
        # First layer (also the only layer if num_layers=1)
        self.convs.append(NodeEdgeInteraction_node(
            in_node_channels,
            in_edge_channels,
            hidden_channels[0] if num_layers > 1 else out_node_channels,
            hidden_channels[0] if num_layers > 1 else out_edge_channels,
            p=dropout if num_layers > 1 else 0  # No dropout for the last layer
        ))
        self.bns.append(nn.BatchNorm1d(hidden_channels[0] if num_layers > 1 else out_node_channels))

        # Hidden layers (only if num_layers > 1)
        for i in range(1, num_layers - 1):
            self.convs.append(NodeEdgeInteraction_node(
                hidden_channels[i - 1],
                hidden_channels[i - 1],
                hidden_channels[i],
                hidden_channels[i],
                p=dropout
            ))
            self.bns.append(nn.BatchNorm1d(hidden_channels[i]))

        # Output layer (only if num_layers > 1)
        if num_layers > 1:
            self.convs.append(NodeEdgeInteraction_node(
                hidden_channels[-1],
                hidden_channels[-1],
                out_node_channels,
                out_edge_channels,
                p=0  # No dropout for the last layer
            ))
            self.bns.append(nn.BatchNorm1d(out_node_channels))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        for i in range(self.num_layers):
            x, edge_attr = self.convs[i](x, edge_index, edge_attr)
            x = self.bns[i](x)
        return x
class EdgeDecoder(nn.Module):
    """Simple MLP Edge Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,
        num_layers=2, dropout=0.5, activation='relu'
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        # 第一层
        self.mlps.append(nn.Linear(in_channels, hidden_channels[0]))
        # 中间层
        for i in range(1, num_layers-1):
            self.mlps.append(nn.Linear(hidden_channels[i - 1], hidden_channels[i]))
        # 输出层
        self.mlps.append(nn.Linear(hidden_channels[-1], out_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, z,edges, sigmoid=True, reduction=False):
        x = z[edges[0]] * z[edges[1]]

        if reduction:
            x = x.mean(-1).unsqueeze(-1)
        x = x.view(-1, x.size(-1))
        for i, mlp in enumerate(self.mlps[:-1]):
            if self.training:  # 判断是否在训练模式下
                x = self.dropout(x)
            x = mlp(x)
            x = self.activation(x)

        x = self.mlps[-1](x)

        if sigmoid:
            x = x.sigmoid()
        return x

        # return x.view(node_num, node_num)
class DegreeDecoder(nn.Module):
    """Simple MLP Degree Decoder"""

    def __init__(
        self, in_channels, hidden_channels, out_channels=1,num_layers=2, dropout=0.5
    ):

        super().__init__()
        self.mlps = nn.ModuleList()

        # 第一层
        self.mlps.append(nn.Linear(in_channels, hidden_channels[0]))

        # 中间层
        for i in range(1, num_layers-1):
            self.mlps.append(nn.Linear(hidden_channels[i - 1], hidden_channels[i]))

        # 输出层
        self.mlps.append(nn.Linear(hidden_channels[-1], out_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, x):

        for i, mlp in enumerate(self.mlps[:-1]):
            x = mlp(x)
            if self.training:  # 判断是否在训练模式下
                x = self.dropout(x)
            x = self.activation(x)

        x = self.mlps[-1](x)
        x = self.activation(x)

        return x

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, modality_a, modality_b):
        """
        模态 A 的特征通过模态 B 的特征进行增强
        Args:
            modality_a: Tensor of shape (batch_size, node_num, embed_dim)
            modality_b: Tensor of shape (batch_size, node_num, embed_dim)

        Returns:
            Tensor of shape (batch_size, node_num, embed_dim)
        """
        # Apply cross-attention
        # Query: modality_a, Key & Value: modality_b
        attn_output, _ = self.multihead_attn(query=modality_a, key=modality_b, value=modality_b)

        output = self.layer_norm(attn_output)
        return output
class AbetaDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1, num_layers=3, dropout=0.5):
        super(AbetaDecoder, self).__init__()
        self.mlps = nn.ModuleList()
        # 第一层
        self.mlps.append(nn.Linear(in_channels, hidden_channels[0]))
        # 中间层
        for i in range(1, num_layers-1):
            self.mlps.append(nn.Linear(hidden_channels[i-1], hidden_channels[i]))
        # 输出层
        self.mlps.append(nn.Linear(hidden_channels[-1], out_channels))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def reset_parameters(self):
        for mlp in self.mlps:
            mlp.reset_parameters()

    def forward(self, x,return_MLP_feat=0):
        if return_MLP_feat:
            MLP_feat = []
        for i, mlp in enumerate(self.mlps[:-1]):
            x = self.dropout(x) if self.training else x
            x = mlp(x)
            x = self.activation(x)
            if return_MLP_feat:
                MLP_feat.append(x)

        x = self.mlps[-1](x)
        if return_MLP_feat:
            MLP_feat.append(x)
            return x,MLP_feat
        return x
class MaskGAE_stage1(nn.Module):
    def __init__(self,
                 encoder_hidden_channels = [16],
                 encoder_node_out_channels  = 4,
                 encoder_edge_out_channels  = 4,
                 encoder_num_layers = 3,
                 encoder_dropout = 0.5,
                 edge_decoder_hidden_channels=[64, 128,32],
                 edge_decoder_num_layers=4,
                 edge_decoder_dropout=0.5,
                 degree_decoder_hidden_channels=[64,128,32],
                 degree_decoder_num_layers=4,
                 degree_decoder_dropout=0.5,
                 decoder_type ='MLP',
                 cross_fusion_type='attn',# attn,cat
                 deg = False,
                 num_heads = 1
                ):
        super(MaskGAE_stage1, self).__init__()
        self.fc_encoder = GNNEncoder(16,
                                     1,
                                     encoder_hidden_channels,
                                     encoder_node_out_channels,
                                     encoder_edge_out_channels,
                                     encoder_num_layers,
                                     encoder_dropout)
        self.sc_encoder = GNNEncoder(3,
                                     3,
                                     encoder_hidden_channels,
                                     encoder_node_out_channels,
                                     encoder_edge_out_channels,
                                     encoder_num_layers,
                                     encoder_dropout)
        self.cross_fussion_type = cross_fusion_type
        if cross_fusion_type == 'attn':
            self.fc_cross_attn = CrossAttention(embed_dim=encoder_node_out_channels, num_heads=num_heads)
            self.sc_cross_attn = CrossAttention(embed_dim=encoder_node_out_channels, num_heads=num_heads)

        if decoder_type == 'dot':
            self.fc_edge_decoder = DotEdgeDecoder()
            self.sc_edge_decoder = DotEdgeDecoder()
        else:
            self.fc_edge_decoder = EdgeDecoder(encoder_node_out_channels*2,
                                            hidden_channels=edge_decoder_hidden_channels,
                                            out_channels=1,
                                            num_layers=edge_decoder_num_layers,
                                            dropout=edge_decoder_dropout)
            self.sc_edge_decoder = EdgeDecoder(encoder_node_out_channels*2,
                                               hidden_channels=edge_decoder_hidden_channels,
                                               out_channels=1,
                                               num_layers=edge_decoder_num_layers,
                                               dropout=edge_decoder_dropout)
        if deg:
            self.fc_degree_decoder = DegreeDecoder(encoder_node_out_channels*2,
                                                degree_decoder_hidden_channels,
                                                out_channels=1,
                                                num_layers=degree_decoder_num_layers,
                                                dropout=degree_decoder_dropout)
            self.sc_degree_decoder = DegreeDecoder(encoder_node_out_channels*2,
                                                   degree_decoder_hidden_channels,
                                                   out_channels=1,
                                                   num_layers=degree_decoder_num_layers,
                                                   dropout=degree_decoder_dropout)
        else:
            self.degree_decoder = None

        self.reset_parameters()

    def reset_parameters(self):
        self.fc_edge_decoder.reset_parameters()
        self.sc_edge_decoder.reset_parameters()
        if self.degree_decoder is not None:
            self.fc_degree_decoder.reset_parameters()
            self.sc_degree_decoder.reset_parameters()


    def forward(self,fc_graph,sc_graph=None):
        batch_size = fc_graph.batch_size
        fc_x = fc_graph.x
        fc_edge_index = fc_graph.edge_index
        fc_mask = fc_graph.edge_mask
        fc_remaining_edges = fc_edge_index[:, fc_mask == 0]
        fc_remaining_edge_attrs = fc_graph.edge_attr[fc_mask == 0]
        fc_mased_edges = fc_edge_index[:, fc_mask == 1]
        fc_neg_mask = fc_graph.neg_mask
        fc_neg_edges = fc_edge_index[:, fc_neg_mask == 1]
        fc_z = self.fc_encoder(fc_x, fc_remaining_edges, fc_remaining_edge_attrs)

        sc_x = sc_graph.x
        sc_edge_index = sc_graph.edge_index
        sc_mask = sc_graph.edge_mask
        sc_remaining_edges = sc_edge_index[:, sc_mask == 0]
        sc_remaining_edge_attrs = sc_graph.edge_attr[sc_mask == 0]
        sc_mased_edges = sc_edge_index[:, sc_mask == 1]
        sc_neg_mask = sc_graph.neg_mask
        sc_neg_edges = sc_edge_index[:, sc_neg_mask == 1]
        sc_z = self.sc_encoder(sc_x, sc_remaining_edges, sc_remaining_edge_attrs)

        if self.cross_fussion_type == 'attn':
            fc_z = fc_z.view(batch_size, -1, fc_z.size(-1))
            sc_z = sc_z.view(batch_size, -1, sc_z.size(-1))
            fc_enhance_z = self.fc_cross_attn(fc_z, sc_z)
            sc_enhance_z = self.sc_cross_attn(sc_z, fc_z)
            z_concat = torch.cat((fc_enhance_z, sc_enhance_z), dim=-1)
        else:
            z_concat = torch.cat((fc_z, sc_z), dim=-1)
        z_concat = z_concat.view(z_concat.size(0)*z_concat.size(1), -1)


        fc_pos_out = self.fc_edge_decoder(z_concat, fc_mased_edges, sigmoid=True)
        fc_neg_out = self.fc_edge_decoder(z_concat, fc_neg_edges, sigmoid=True)
        fc_deg, fc_deg_out = None, None
        if self.degree_decoder is not None:
            fc_deg = degree(fc_mased_edges[1], fc_graph.num_nodes).float()
            fc_deg_out = self.degree_decoder(fc_z).squeeze()

        sc_pos_out = self.sc_edge_decoder(z_concat, sc_mased_edges, sigmoid=True)
        sc_neg_out = self.sc_edge_decoder(z_concat, sc_neg_edges, sigmoid=True)
        sc_deg, sc_deg_out = None, None
        if self.degree_decoder is not None:
            sc_deg = degree(sc_mased_edges[1], sc_graph.num_nodes).float()
            sc_deg_out = self.degree_decoder(sc_z).squeeze()


        return fc_pos_out,fc_neg_out,fc_deg,fc_deg_out,sc_pos_out,sc_neg_out,sc_deg,sc_deg_out
class MaskGAE_stage2(nn.Module):
    def __init__(self,
                 encoder_hidden_channels = [16],
                 encoder_node_out_channels  = 4,
                 encoder_edge_out_channels  = 4,
                 encoder_num_layers = 3,
                 encoder_dropout = 0.5,
                 abeta_decoder_hidden_channels=[64, 128,32],
                 abeta_decoder_num_layers=4,
                 abeta_decoder_dropout=0.5,
                 cross_fusion_type='attn',# attn,cat
                 num_heads = 1
                ):
        super(MaskGAE_stage2, self).__init__()
        self.fc_encoder = GNNEncoder(16,
                                     1,
                                     encoder_hidden_channels,
                                     encoder_node_out_channels,
                                     encoder_edge_out_channels,
                                     encoder_num_layers,
                                     encoder_dropout)
        self.sc_encoder = GNNEncoder(3,
                                     3,
                                     encoder_hidden_channels,
                                     encoder_node_out_channels,
                                     encoder_edge_out_channels,
                                     encoder_num_layers,
                                     encoder_dropout)
        self.cross_fussion_type = cross_fusion_type
        if cross_fusion_type == 'attn':
            self.fc_cross_attn = CrossAttention(embed_dim=encoder_node_out_channels, num_heads=num_heads)
            self.sc_cross_attn = CrossAttention(embed_dim=encoder_node_out_channels, num_heads=num_heads)

        self.abeta_decoder = AbetaDecoder(in_channels=encoder_node_out_channels *2* 100,  # *2 是因为要concat两个encoder的输出
                                          hidden_channels=abeta_decoder_hidden_channels,
                                          out_channels=100,
                                          num_layers=abeta_decoder_num_layers,
                                          dropout=abeta_decoder_dropout)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc_encoder.reset_parameters()
        self.sc_encoder.reset_parameters()
        self.abeta_decoder.reset_parameters()


    def forward(self,fc_graph,sc_graph=None,return_attention_feature=False,return_MLP_feat=2):
        batch_size = fc_graph.batch_size
        fc_x = fc_graph.x
        fc_edge_index = fc_graph.edge_index
        fc_remaining_edge_attrs = fc_graph.edge_attr
        fc_z = self.fc_encoder(fc_x, fc_edge_index, fc_remaining_edge_attrs)

        sc_x = sc_graph.x
        sc_edge_index = sc_graph.edge_index
        sc_remaining_edge_attrs = sc_graph.edge_attr
        sc_z = self.sc_encoder(sc_x, sc_edge_index, sc_remaining_edge_attrs)

        if self.cross_fussion_type == 'attn':
            fc_z = fc_z.view(batch_size, -1, fc_z.size(-1))
            sc_z = sc_z.view(batch_size, -1, sc_z.size(-1))
            fc_enhance_z = self.fc_cross_attn(fc_z, sc_z)
            sc_enhance_z = self.sc_cross_attn(sc_z, fc_z)
            z_concat = torch.cat((fc_enhance_z, sc_enhance_z), dim=-1)
        else:
            z_concat = torch.cat((fc_z, sc_z), dim=-1)
        z_concat = z_concat.view(batch_size, -1)
        MLP_feat = None
        if return_MLP_feat:
            abeta_out,MLP_feat = self.abeta_decoder(z_concat,return_MLP_feat=return_MLP_feat)
        else:
            abeta_out = self.abeta_decoder(z_concat,return_MLP_feat=return_MLP_feat)
        if return_attention_feature:
            if return_MLP_feat:
                return abeta_out,z_concat,MLP_feat
            return abeta_out,z_concat
        return abeta_out
