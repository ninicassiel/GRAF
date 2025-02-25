# -*- coding:UTF-8 -*-
'''
@Date   :{DATE}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_loss_fn(loss_type):
    """
        根据loss_type选择合适的损失函数
        :param loss_type: 'acc' 或 'bceloss'
        :return: 对应的损失函数
        """
    if loss_type == "acc":
        return accuracy_loss
    elif loss_type == "bce":
        return bce_loss
    elif loss_type == "l1":
        return l1_loss
    elif loss_type=='focal':
        return FocalLoss()
    else:
        raise ValueError(f"不支持的损失类型: {loss_type}")

def accuracy_loss(pos_out, neg_out,pos_weight=None):
    """
    计算正负样本的准确率
    :param pos_out: 正样本预测的logits或概率
    :param neg_out: 负样本预测的logits或概率
    :return: 准确率 (accuracy)
    """
    pos_correct = (pos_out >= 0.5).float().mean()  # 正样本中预测为1的比例
    neg_correct = (neg_out < 0.5).float().mean()  # 负样本中预测为0的比例
    if pos_weight is not None:
        acc = (pos_correct * pos_weight + neg_correct) / (pos_weight+1)
    else:
        acc = (pos_correct + neg_correct) / 2.0

    return 1 - acc

def bce_loss(pos_out,neg_out, pos_weight=None):
    """
    计算二分类交叉熵损失 (BCEWithLogitsLoss)
    :param pos_out: 正样本预测的logits
    :param neg_out: 负样本预测的logits
    :param pos_weight: 正样本的权重 (用于处理不平衡数据)
    :return: BCE损失
    """
    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight).to(pos_out.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 创建正负样本的标签
    pos_labels = torch.ones_like(pos_out)  # 正样本的标签为1
    neg_labels = torch.zeros_like(neg_out)  # 负样本的标签为0

    # 拼接正样本和负样本的输出及标签
    all_out = torch.cat([pos_out, neg_out])
    all_labels = torch.cat([pos_labels, neg_labels])

    # # 计算 BCEWithLogitsLoss 损失
    loss = loss_fn(all_out, all_labels)
    return loss
def l1_loss(pos_out,neg_out, pos_weight=None):
    # 创建正负样本的标签
    pos_labels = torch.ones_like(pos_out)  # 正样本的标签为1
    neg_labels = torch.zeros_like(neg_out)  # 负样本的标签为0

    # 拼接正样本和负样本的输出及标签
    all_out = torch.cat([pos_out, neg_out])
    all_labels = torch.cat([pos_labels, neg_labels])
    return F.l1_loss(all_out, all_labels)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.2, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pos_out,neg_out, pos_weight=None):
        # 创建正负样本的标签
        pos_labels = torch.ones_like(pos_out)  # 正样本的标签为1
        neg_labels = torch.zeros_like(neg_out)  # 负样本的标签为0

        # 拼接正样本和负样本的输出及标签
        all_out = torch.cat([pos_out, neg_out])
        all_labels = torch.cat([pos_labels, neg_labels])

        BCE_loss = F.binary_cross_entropy_with_logits(all_out, all_labels, reduction='none')
        pt = torch.exp(-BCE_loss)  # 计算预测的概率
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss