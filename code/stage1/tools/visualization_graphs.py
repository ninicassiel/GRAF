# -*- coding:UTF-8 -*-
'''
用于比较原始图和恢复图
@Date   :{DATE}
'''

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings("ignore")


def visualize_edge(pos_edge_index, neg_edge_index,
                   restored_pos_edge, restored_neg_edge,
                   num_node,
                   save_path=None):
    '''
    original_edge_index: 存在的正样本对（2，edge_num）
    mask_edge_index: 不存在的负样本对（2，edge_num）
    restored_pos_edge: 预测的正样edge存在的概率 (edge_num,)
    restored_neg_edge: 预测的负样本edge存在的概率 (edge_num,)
    num_node: 节点数量
    '''

    # 创建原始图
    original_graph = nx.Graph()
    for edge in pos_edge_index.T:
        original_graph.add_edge(edge[0], edge[1])

    # 创建恢复图
    restored_graph = nx.Graph()

    # 定义边的分类
    TP = pos_edge_index[:, restored_pos_edge >= 0.5]
    FN = pos_edge_index[:, restored_pos_edge < 0.5]
    TN = neg_edge_index[:, restored_neg_edge < 0.5]
    FP = neg_edge_index[:, restored_neg_edge >= 0.5]


    # 添加正样本边
    for i, prob in enumerate(pos_edge_index):
        restored_graph.add_edge(pos_edge_index[0, i], pos_edge_index[1, i])

    # 添加负样本边
    for i, prob in enumerate(neg_edge_index):
        restored_graph.add_edge(neg_edge_index[0,i], neg_edge_index[1, i])
    # 确保所有节点都存在
    if len(restored_graph.nodes) < num_node:
        for node in range(num_node):
            restored_graph.add_node(node)
    # 绘制图
    plt.figure(figsize=(12, 6))

    # 原始图
    plt.subplot(1, 2, 1)
    nx.draw(original_graph, with_labels=True, edge_color='blue', node_color='lightgray')
    plt.title('Original Graph')

    # 恢复图
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(restored_graph)  # 生成布局

    # 绘制TP和FN（同一色系，绿色）
    if TP.shape[1] > 0:  # 确保TP不为空
        nx.draw_networkx_edges(restored_graph, pos=pos, edgelist=TP.T.tolist(),
                               edge_color='green', width=2, label='True Positives')
    if FN.shape[1] > 0:  # 确保FN不为空
        nx.draw_networkx_edges(restored_graph, pos=pos, edgelist=FN.T.tolist(),
                               edge_color='lightgreen', width=2, label='False Negatives', style='dashed')

    # 绘制TN和FP（同一色系，红色）
    if TN.shape[1] > 0:  # 确保TN不为空
        nx.draw_networkx_edges(restored_graph, pos=pos, edgelist=TN.T.tolist(),
                               edge_color='red', width=2, label='True Negatives')
    if FP.shape[1] > 0:  # 确保FP不为空
        nx.draw_networkx_edges(restored_graph, pos=pos, edgelist=FP.T.tolist(),
                               edge_color='salmon', width=2, label='False Positives', style='dashed')

    # 绘制节点
    nx.draw_networkx_nodes(restored_graph, pos=pos, node_color='lightgray')
    plt.title('Restored Graph')
    plt.legend()

    if save_path:
            plt.savefig(save_path, dpi=500)
    plt.show()


def visualize_hitmap(pos_edge_index, neg_edge_index,remain_edge_index,
                     restored_pos_edge, restored_neg_edge,
                     num_node, save_path=None,show=True):
    # 创建原始、恢复和错误预测的邻接矩阵
    original_adj_matrix = np.zeros((num_node, num_node))  # 默认为0（灰色）
    restored_adj_matrix = np.zeros((num_node, num_node))
    error_adj_matrix = np.zeros((num_node, num_node))

    # 定义边的分类
    TP = pos_edge_index[:, restored_pos_edge >= 0.5]  # 正样本正确预测 (True Positive)
    FN = pos_edge_index[:, restored_pos_edge < 0.5]  # 正样本错误预测 (False Negative)
    TN = neg_edge_index[:, restored_neg_edge < 0.5]  # 负样本正确预测 (True Negative)
    FP = neg_edge_index[:, restored_neg_edge >= 0.5]  # 负样本错误预测 (False Positive)

    # 填充原始邻接矩阵
    for edge in pos_edge_index.T:
        original_adj_matrix[edge[0], edge[1]] = 1  # 正样本边（红色）
    is_symmetric = np.array_equal(original_adj_matrix, original_adj_matrix.T)
    print(f"Original adjacency matrix (pos) is symmetric: {is_symmetric}")
    for edge in neg_edge_index.T:
        original_adj_matrix[edge[0], edge[1]] = -1  # 负样本边（蓝色）
    is_symmetric = np.array_equal(original_adj_matrix, original_adj_matrix.T)
    print(f"Original adjacency matrix (neg) is symmetric: {is_symmetric}")

    for edge in remain_edge_index.T:
        original_adj_matrix[edge[0], edge[1]] = 0.5  # 剩余样本边（绿色）
    is_symmetric = np.array_equal(original_adj_matrix, original_adj_matrix.T)
    print(f"Original adjacency matrix (remain) is symmetric: {is_symmetric}")

    # 恢复矩阵填充：TP -> 1, TN -> -1
    for edge in TP.T:
        restored_adj_matrix[edge[0], edge[1]] = 1
    for edge in TN.T:
        restored_adj_matrix[edge[0], edge[1]] = -1

    # 错误预测矩阵填充：FN -> -1, FP -> 1
    for edge in FN.T:
        error_adj_matrix[edge[0], edge[1]] = -1
    for edge in FP.T:
        error_adj_matrix[edge[0], edge[1]] = 1
    colors = ['#7E9bb7', 'white','#F9E9A4','#F89FA8']
    colors = ['blue','white','yellow','red']
    # 自定义颜色映射：正样本用红色，负样本用蓝色，剩余样本用绿色，未定义区域用白色
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=[-1.5, -0.5, 0.25, 0.75, 1.5], ncolors=cmap.N)
    cmap2 = ListedColormap([colors[0], 'white',colors[-1]])

    plt.figure(figsize=(12, 4))

    # 绘制原始邻接矩阵
    plt.subplot(1, 3, 1)
    ax = sns.heatmap(original_adj_matrix, cmap=cmap,norm=norm, cbar=False, annot=False)
    plt.title('Original Adjacency Matrix')
    plt.xticks([], [])
    plt.yticks([], [])
    # ax.set_facecolor('black')
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor('black')
    # ax.patch.set_edgecolor('black')
    # ax.patch.set_linewidth(2)

    # 绘制恢复邻接矩阵
    plt.subplot(1, 3, 2)
    ax = sns.heatmap(restored_adj_matrix, cmap=cmap2, cbar=False, annot=False, center=0, vmin=-1, vmax=1)
    plt.title('Restored Adjacency Matrix')
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_facecolor('black')
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor('black')
    # ax.patch.set_edgecolor('black')
    # ax.patch.set_linewidth(2)

    # 绘制错误预测矩阵
    plt.subplot(1, 3, 3)
    ax = sns.heatmap(error_adj_matrix, cmap=cmap2, cbar=False, annot=False, center=0, vmin=-1, vmax=1)
    plt.title('Error Adjacency Matrix')
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_facecolor('black')
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor('black')
    # ax.patch.set_edgecolor('black')
    # ax.patch.set_linewidth(2)

    # 添加图例
    red_patch = mpatches.Patch(color=colors[-1], label='Positive', edgecolor='black')
    blue_patch = mpatches.Patch(color=colors[0], label='Negative', edgecolor='black')
    green_patch = mpatches.Patch(color=colors[-2], label='Remain', edgecolor='black')

    plt.legend(handles=[red_patch, blue_patch, green_patch], bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0.)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=500)
    if show:
        plt.show()



def visualize_hitmap_v2(pos_edge_index, neg_edge_index,remain_edge_index,
                     restored_pos_edge, restored_neg_edge,
                     num_node, save_path=None,show=True):
    # 创建原始、恢复和错误预测的邻接矩阵
    original_adj_matrix = np.zeros((num_node, num_node))
    targget_adj_matrix = np.zeros((num_node, num_node))
    restored_adj_matrix = np.zeros((num_node, num_node))
    error_adj_matrix = np.zeros((num_node, num_node))

    # 定义边的分类
    TP = pos_edge_index[:, restored_pos_edge >= 0.5]  # 正样本正确预测 (True Positive)
    FN = pos_edge_index[:, restored_pos_edge < 0.5]  # 正样本错误预测 (False Negative)
    TN = neg_edge_index[:, restored_neg_edge < 0.5]  # 负样本正确预测 (True Negative)
    FP = neg_edge_index[:, restored_neg_edge >= 0.5]  # 负样本错误预测 (False Positive)

    # 填充原始邻接矩阵
    for edge in pos_edge_index.T:
        original_adj_matrix[edge[0], edge[1]] = 1  # 正样本边（红色）
        targget_adj_matrix[edge[0], edge[1]] = 1  # 正样本边（红色）
    is_symmetric = np.array_equal(original_adj_matrix, original_adj_matrix.T)
    print(f"Original adjacency matrix (pos) is symmetric: {is_symmetric}")
    for edge in neg_edge_index.T:
        original_adj_matrix[edge[0], edge[1]] = -1  # 负样本边（蓝色）
        targget_adj_matrix[edge[0], edge[1]] = -1  # 负样本边（蓝色）
    is_symmetric = np.array_equal(original_adj_matrix, original_adj_matrix.T)
    print(f"Original adjacency matrix (neg) is symmetric: {is_symmetric}")

    for edge in remain_edge_index.T:
        original_adj_matrix[edge[0], edge[1]] = 0.5  # 剩余样本边（绿色）
    is_symmetric = np.array_equal(original_adj_matrix, original_adj_matrix.T)
    print(f"Original adjacency matrix (remain) is symmetric: {is_symmetric}")

    # 恢复矩阵填充：TP -> 1, TN -> -1
    for edge in TP.T:
        restored_adj_matrix[edge[0], edge[1]] = 1
    for edge in TN.T:
        restored_adj_matrix[edge[0], edge[1]] = -1

    # 错误预测矩阵填充：FN -> -1, FP -> 1
    for edge in FN.T:
        error_adj_matrix[edge[0], edge[1]] = -1
        restored_adj_matrix[edge[0], edge[1]] = -1
    for edge in FP.T:
        error_adj_matrix[edge[0], edge[1]] = 1
        restored_adj_matrix[edge[0], edge[1]] = 1
    colors = ['blue','white','yellow','red']
    # 自定义颜色映射：正样本用红色，负样本用蓝色，剩余样本用绿色，未定义区域用白色
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=[-1.5, -0.5, 0.25, 0.75, 1.5], ncolors=cmap.N)
    cmap2 = ListedColormap([colors[0], 'white',colors[-1]])

    plt.figure(figsize=(12, 8))

    # 绘制原始邻接矩阵
    plt.subplot(2, 2, 1)
    ax = sns.heatmap(original_adj_matrix, cmap=cmap,square=True,norm=norm, cbar=False, annot=False)
    plt.title('Original Adjacency Matrix')
    plt.xticks([], [])
    plt.yticks([], [])
    # ax.set_facecolor('black')
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor('black')

    # 绘制原始邻接矩阵
    plt.subplot(2, 2, 2)
    ax = sns.heatmap(targget_adj_matrix, cmap=cmap2,square=True, cbar=False, annot=False, center=0, vmin=-1, vmax=1)
    plt.title('Target Adjacency Matrix')
    plt.xticks([], [])
    plt.yticks([], [])
    # ax.set_facecolor('black')
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor('black')

    # 绘制恢复邻接矩阵
    plt.subplot(2, 2, 3)
    ax = sns.heatmap(restored_adj_matrix, cmap=cmap2,square=True, cbar=False, annot=False, center=0, vmin=-1, vmax=1)
    plt.title('Restored Adjacency Matrix')
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_facecolor('black')
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor('black')
    # ax.patch.set_edgecolor('black')
    # ax.patch.set_linewidth(2)

    # 绘制错误预测矩阵
    plt.subplot(2, 2, 4)
    ax = sns.heatmap(error_adj_matrix, cmap=cmap2,square=True, cbar=False, annot=False, center=0, vmin=-1, vmax=1)
    plt.title('Error Adjacency Matrix')
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_facecolor('black')
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor('black')
    # ax.patch.set_edgecolor('black')
    # ax.patch.set_linewidth(2)

    # 添加图例
    red_patch = mpatches.Patch(color=colors[-1], label='Positive', edgecolor='black')
    blue_patch = mpatches.Patch(color=colors[0], label='Negative', edgecolor='black')
    green_patch = mpatches.Patch(color=colors[-2], label='Remain', edgecolor='black')

    plt.legend(handles=[red_patch, blue_patch, green_patch], bbox_to_anchor=(1.02, 1),
               loc='upper left',
               frameon=False,
               borderaxespad=0.)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=500)
    if show:
        plt.show()

def visualiaze_mask_hitmap(pos_edge_index, neg_edge_index,remain_edge_index,num_node,
                           save_path=None,show=True):
    original_adj_matrix = np.zeros((num_node, num_node))  # 默认为0（灰色）
    for edge in pos_edge_index.T:
        original_adj_matrix[edge[0], edge[1]] = 1  # 正样本边（红色）
    for edge in neg_edge_index.T:
        original_adj_matrix[edge[0], edge[1]] = -1
    for edge in remain_edge_index.T:
        original_adj_matrix[edge[0], edge[1]] = 0.5
    colors = ['blue', 'white', 'yellow', 'red']
    # 自定义颜色映射：正样本用红色，负样本用蓝色，剩余样本用绿色，未定义区域用白色
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(boundaries=[-1.5, -0.5, 0.25, 0.75, 1.5], ncolors=cmap.N)

    plt.figure(figsize=(8, 6))

    # 绘制邻接矩阵
    ax = sns.heatmap(original_adj_matrix, cmap=cmap, norm=norm, cbar=False, annot=False)
    plt.title('Adjacency Matrix with mask')
    plt.xticks([], [])
    plt.yticks([], [])
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_edgecolor('black')

    # 创建图例
    red_patch = mpatches.Patch(color=colors[-1], label=f'Makse{pos_edge_index.size(1)}', edgecolor='black')
    blue_patch = mpatches.Patch(color=colors[0], label=f'Negative{neg_edge_index.size(1)}', edgecolor='black')
    green_patch = mpatches.Patch(color=colors[-2], label=f'Unmaske{remain_edge_index.size(1)}', edgecolor='black')

    # 调整图例位置
    plt.legend(handles=[red_patch, blue_patch, green_patch], bbox_to_anchor=(1.05, 1), loc='upper left',
               borderaxespad=0.)

    # 调整布局以确保图例在图内
    plt.tight_layout(rect=[0, 0, 0.9, 0.9])

    if save_path is not None:
        plt.savefig(save_path, dpi=500)
    if show:
        plt.show()




