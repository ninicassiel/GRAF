# -*- coding:UTF-8 -*-
'''
@Date   :{DATE}
'''
import os

import numpy as np
import scipy.io as sio
from tools.visualization_graphs import visualize_hitmap,visualize_hitmap_v2

#### version 1 #####
data_dir = r'D:\Projects\Output\dmri_fmri2PET\PyG_models_v8_2\2024_11_24_15_43_31_sc_0.4_0.2\fold_5\validation_epoch992_ACC0.7363_0.0495_AUC0.8008_0.0603'
sub_files = ['AFM0320_Acc0.7784_AUC0.8603_Sensitivity0.7592_Specificity0.7977.mat',
            'AFM0526_Acc0.8161_AUC0.8811_Sensitivity0.7876_Specificity0.8445.mat',
             'AFM0571_Acc0.8094_AUC0.8684_Sensitivity0.7926_Specificity0.8261.mat',
             'AFM0116_Acc0.6572_AUC0.6844_Sensitivity0.7726_Specificity0.5418.mat',
             'AFM0180_Acc0.6329_AUC0.7023_Sensitivity0.8729_Specificity0.3930.mat'
             ]
# for sub_file in os.listdir(data_dir):
for sub_file in sub_files:
    if not sub_file.endswith('mat'):
        continue

    data_path = os.path.join(data_dir, sub_file)

# data_path = 'D:/Projects/Output/dmri_fmri2PET/PyG_models/part_edge_2024_09_25_22_18_58fc/fold_2/validation/AFM0006.mat'
    data = sio.loadmat(data_path)
    mask_edge = data['mask_edge']
    neg_edge = data['neg_edge']
    remain_edge = data['remain_edge']
    mase_shape = mask_edge.shape
    neg_shape = neg_edge.shape
    remain_shape = remain_edge.shape
    # if abs(neg_shape[-1]-mase_shape[-1])>30:10
    print(sub_file)
    # break

    print(mask_edge.shape)
    print(neg_edge.shape)
    print(remain_edge.shape)

    pos_out = np.squeeze(data['pos_out'])
    neg_out = np.squeeze(data['neg_out'])
    num_node = 100
    save_path = os.path.join(data_dir, sub_file.replace('.mat', '.png'))
    visualize_hitmap_v2(mask_edge, neg_edge,remain_edge,
                     pos_out, neg_out, num_node,save_path,show=False)
    # break

##### version 2 #####
# data_dir = 'D:/Projects/Output/dmri_fmri2PET/PyG_models_v3/test/fold_5/validation'
# for sub_file in os.listdir(data_dir):
#     if not sub_file.endswith('mat'):
#         continue
#     data_path = os.path.join(data_dir, sub_file)
#     data = sio.loadmat(data_path)
#     # mask_edge = data['mask']
#     # neg_edge = data['neg_edge']
#     # remain_edge = data['remain_edge']
#     # mase_shape = mask_edge.shape
#     # neg_shape = neg_edge.shape
#     # remain_shape = remain_edge.shape
#     mask = data['mask']
#     pred = data['pred']
#     label = data['label']
#     # if abs(neg_shape[-1]-mase_shape[-1])>30:10
#     print(sub_file)
#     # break
#     #
#     # print(mask_edge.shape)
#     # print(neg_edge.shape)
#     # print(remain_edge.shape)
#
#     # pos_out = np.squeeze(data['pos_out'])
#     # neg_out = np.squeeze(data['neg_out'])
#     # num_node = 100
#     save_path = os.path.join(data_dir, sub_file.replace('.mat', '.png'))
#     visualize_hitmap_v2(mask= mask,label = label,pred = pred,save_path = save_path,show=False)
#     # break
