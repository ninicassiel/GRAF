# -*- coding:UTF-8 -*-
'''
@Date   :{DATE}
'''
import torch
import torch.nn.functional as F

def auc_loss(pos_out, neg_out):
    return torch.square(1 - (pos_out - neg_out)).sum()

def hinge_auc_loss(pos_out, neg_out):
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()

def log_rank_loss(pos_out, neg_out, num_neg=1):
    return -torch.log(torch.sigmoid(pos_out - neg_out) + 1e-15).mean()

def ce_loss(pos_out, neg_out):
    # # 样本数量
    # N_pos = pos_out.size(0)
    # N_neg = neg_out.size(0)
    #
    # # 计算正负样本的权重比例
    # total = N_pos + N_neg
    # pos_weight = N_neg / total
    # neg_weight = N_pos / total
    total_loss = 0
    if pos_out!=None:
        # 计算正样本和负样本的损失
        pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
        total_loss+=pos_loss
    if neg_out!=None:
        neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
        total_loss+=neg_loss

    # 对正负样本损失进行加权相加
    # total_loss = pos_weight * pos_loss + neg_weight * neg_loss
    # total_loss = pos_loss + neg_loss
    return total_loss

def info_nce_loss(pos_out, neg_out):
    pos_exp = torch.exp(pos_out)
    neg_exp = torch.sum(torch.exp(neg_out), 1, keepdim=True)
    return -torch.log(pos_exp / (pos_exp + neg_exp) + 1e-15).mean()