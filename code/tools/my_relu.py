# -*- coding:UTF-8 -*-
'''
@Date   :{DATE}
'''
import torch
def my_relu(x):
    return torch.maximum(x, torch.zeros_like(x))