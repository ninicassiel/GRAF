# -*- coding:UTF-8 -*-
'''
@Date   :{DATE}
'''
import sys
import os
project_root = os.path.abspath(r"D:/Projects/p#001DTI2PET/dmri_fmri2PET")
if project_root not in sys.path:
    sys.path.append(project_root)
import numpy as np
import torch
import random
import json
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from functools import partial
import torch
import pandas as pd

from mymodels.PyG_tools.dataset import MyGraphDataset, collate_fn, MyGraphDataset_withPET

from models import MaskGAE_stage1,MaskGAE_stage2
from torch_geometric.data import Batch
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import scipy.io as sio

from mymodels.PyG_tools.evaluator import evaluator,evaluator_v2,evaluator_v3
from mymodels.PyG_tools.save_json import save_args_to_json
import torch.nn as nn
def load_pretrain_model_stage1(pretrain_model_path):
    config_path = os.path.join(os.path.dirname(pretrain_model_path), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        if config['decoder_type']=='mlp':
            pred_model = MaskGAE_stage1(
                encoder_hidden_channels=config['encoder_hidden_channels'],
                encoder_node_out_channels=config['encoder_node_out_channels'],
                encoder_edge_out_channels=config['encoder_edge_out_channels'],
                encoder_num_layers=config['encoder_num_layers'],
                encoder_dropout=config['encoder_dropout'],
                decoder_type=config['decoder_type'],
                edge_decoder_num_layers=config['edge_decoder_num_layers'],
                edge_decoder_hidden_channels=config['edge_decoder_hidden_channels'],
                edge_decoder_dropout=config['edge_decoder_dropout'],
                degree_decoder_hidden_channels=config['degree_decoder_hidden_channels'],
                degree_decoder_num_layers=config['degree_decoder_num_layers'],
                degree_decoder_dropout=config['degree_decoder_dropout'],
                deg=config['deg']
            )
        else:
            pred_model = MaskGAE_stage1(
                encoder_hidden_channels=config['encoder_hidden_channels'],
                encoder_node_out_channels=config['encoder_node_out_channels'],
                encoder_edge_out_channels=config['encoder_edge_out_channels'],
                encoder_num_layers=config['encoder_num_layers'],
                encoder_dropout=config['encoder_dropout'],
                decoder_type=config['decoder_type'],
                degree_decoder_hidden_channels=config['degree_decoder_hidden_channels'],
                degree_decoder_num_layers=config['degree_decoder_num_layers'],
                degree_decoder_dropout=config['degree_decoder_dropout'],
                deg=config['deg'])

        return pred_model
def train_validate(fold,output_dir,args):
    '''
    one modality
    use filter fmri graph or dmri graph, reconstruct the graph (0/1 matrix)
    the num of neg_edge equals to pos_edge
    :param fold:
    :param output_dir:
    :param model_type:
    :return:
    '''
    base_dir = args.base_dir
    table_path = os.path.join(base_dir, args.table_path_file)
    dmri_graph_dir = os.path.join(base_dir, args.dmri_graph_dir)
    fmri_graph_dir = os.path.join(base_dir, args.fmri_graph_dir)
    pet_dir = os.path.join(base_dir, args.pet_dir)

    output_dir = output_dir +  f'/fold_{fold}'
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=f"{output_dir}/logs")
    args.output_dir = output_dir
    save_args_to_json(args, os.path.join(output_dir, 'config.json'))

    ROI = args.ROI
    BOLD = args.BOLD
    lr = args.lr
    weight_decay = args.weight_decay
    epoch = args.epoch
    batch_size = args.batch_size
    fc_threshold = args.fc_threshold
    sc_threshold = args.sc_threshold
    mask_ratio = args.mask_ratio
    model_type = args.model_type

    # 根据 model_type 设置 fc 和 sc
    fc = model_type == 'fc' or model_type == 'both'
    sc = model_type == 'sc' or model_type == 'both'
    if args.minitest:
        train_dataset = MyGraphDataset_withPET(table_path,
                                               dmri_graph_dir,
                                               fmri_graph_dir,
                                               PET_dir=pet_dir,
                                               fc_threshold=fc_threshold, sc_threshold=sc_threshold,
                                               data_type='val', fold=fold, ROI=ROI, BOLD=BOLD)
    else:
        train_dataset = MyGraphDataset_withPET(table_path,
                                               dmri_graph_dir,
                                               fmri_graph_dir,
                                               PET_dir=pet_dir,
                                               fc_threshold=fc_threshold, sc_threshold=sc_threshold,
                                               data_type='train', fold=fold, ROI=ROI, BOLD=BOLD)
    val_dataset = MyGraphDataset_withPET(table_path,
                                         dmri_graph_dir,
                                         fmri_graph_dir,
                                         PET_dir=pet_dir,
                                         fc_threshold=fc_threshold, sc_threshold=sc_threshold,
                                         data_type='val', fold=fold, ROI=ROI, BOLD=BOLD)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=partial(collate_fn,
                                                 mask_ratio=mask_ratio,
                                                 fc=fc,
                                                 sc=sc))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=partial(collate_fn,
                                               mask_ratio=mask_ratio,
                                               fc=fc,
                                               sc=sc))

    pred_model = load_pretrain_model_stage1(args.pretrained).to(device)
    model = MaskGAE_stage2(
        encoder_hidden_channels=args.encoder_hidden_channels,
        encoder_node_out_channels=args.encoder_node_out_channels,
        encoder_edge_out_channels=args.encoder_edge_out_channels,
        encoder_num_layers=args.encoder_num_layers,
        encoder_dropout=args.encoder_dropout,
        abeta_decoder_hidden_channels = args.abeta_decoder_hidden_channels,
        abeta_decoder_num_layers = args.abeta_decoder_num_layers,
        abeta_decoder_dropout = args.abeta_decoder_dropout,
        cross_fusion_type=args.cross_fusion_type,
        num_heads=args.num_heads,
    ).to(device)
    model.fc_encoder.load_state_dict(pred_model.fc_encoder.state_dict())
    model.sc_encoder.load_state_dict(pred_model.sc_encoder.state_dict())
    if args.cross_fusion_type=='attn':
        model.fc_cross_attn.load_state_dict(pred_model.fc_cross_attn.state_dict())
        model.sc_cross_attn.load_state_dict(pred_model.sc_cross_attn.state_dict())

    if args.freeze_encoder:
        for param in model.fc_encoder.parameters():
            param.requires_grad = False
        for param in model.sc_encoder.parameters():
            param.requires_grad = False
        if args.cross_fusion_type == 'attn':
            for param in model.fc_cross_attn.parameters():
                param.requires_grad = False
            for param in model.sc_cross_attn.parameters():
                param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    regre_criterion = nn.MSELoss(reduction='none')
    best_val_regre_loss = float('inf')
    model.train()
    for e in range(epoch):
        model.train()
        total_loss = []
        Regression_loss = []
        for i, batch_data in enumerate(train_loader):
            optimizer.zero_grad()  # 清除之前的梯度
            fc_graphs = batch_data['FC_graphs'].to(device)
            sc_graphs = batch_data['SC_graphs'].to(device)
            pet_label = batch_data['PET_labels'].to(device)
            mask_ratio = mask_ratio

            regression_out = model(fc_graphs, sc_graphs)

            # -------------------处理regression loss
            regression_loss = regre_criterion(regression_out, pet_label).mean(dim=1)
            loss = torch.mean(regression_loss)
            Regression_loss.extend(regression_loss.tolist())

            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        val_Regression_loss = []
        model.eval()

        with torch.no_grad():  # 禁用梯度计算，节省内存和加速验证
            for i, batch_data in enumerate(val_loader):
                fc_graphs = batch_data['FC_graphs'].to(device)
                sc_graphs = batch_data['SC_graphs'].to(device)
                pet_label = batch_data['PET_labels'].to(device)
                regression_out = model(fc_graphs, sc_graphs)

                # -------------------处理regression loss
                regression_loss = regre_criterion(regression_out, pet_label).mean(dim=1)
                val_Regression_loss.extend(regression_loss.tolist())

        writer.add_scalars('Regression_loss', {'train': np.mean(Regression_loss), 'val': np.mean(val_Regression_loss)},
                           e)
        # 用新学的那个模态的指标来更新最优的模型
        val_Regression_loss_mean = np.mean(val_Regression_loss)
        if val_Regression_loss_mean < 0.07 and val_Regression_loss_mean < best_val_regre_loss:
            best_val_regre_loss = val_Regression_loss_mean
            model_path = os.path.join(output_dir,
                                      f"Best_model_epoch_{e:03d}_regreLoss{val_Regression_loss_mean:.4f}.pth")
            torch.save(model.state_dict(), model_path)
        print(
            f'Epoch [{e:03d}] train_regreLoss: {np.mean(Regression_loss):.4f} val_regreLoss: {val_Regression_loss_mean:.4f} ')
    writer.close()

def test_visualize(fold,args,pick):
    '''
    one modality
    use filter fmri graph or dmri graph, reconstruct the graph (0/1 matrix)
    the num of neg_edge equals to pos_edge
    :param fold:
    :param output_dir:
    :param model_type:
    :return:
    '''
    base_dir = args.base_dir
    table_path = os.path.join(base_dir, args.table_path_file)
    dmri_graph_dir = os.path.join(base_dir, args.dmri_graph_dir)
    fmri_graph_dir = os.path.join(base_dir, args.fmri_graph_dir)
    pet_dir = os.path.join(base_dir, args.pet_dir)

    pretrain_model = args.pretrained
    epoch = os.path.split(pretrain_model)[-1].split('_')[3]
    if pick:
        output_dir = f'{os.path.dirname(pretrain_model)}/fold{fold}_validation_epoch{epoch}_pick'
    else:
        output_dir = f'{os.path.dirname(pretrain_model)}/fold{fold}_validation_epoch{epoch}_noPick'
    os.makedirs(output_dir, exist_ok=True)
    ROI = args.ROI
    BOLD = args.BOLD
    batch_size = args.batch_size
    fc_threshold = args.fc_threshold
    sc_threshold = args.sc_threshold
    mask_ratio = args.mask_ratio
    model_type = args.model_type

    model = MaskGAE_stage2(
        encoder_hidden_channels=args.encoder_hidden_channels,
        encoder_node_out_channels=args.encoder_node_out_channels,
        encoder_edge_out_channels=args.encoder_edge_out_channels,
        encoder_num_layers=args.encoder_num_layers,
        encoder_dropout=args.encoder_dropout,
        abeta_decoder_hidden_channels=args.abeta_decoder_hidden_channels,
        abeta_decoder_num_layers=args.abeta_decoder_num_layers,
        abeta_decoder_dropout=args.abeta_decoder_dropout,
        cross_fusion_type=args.cross_fusion_type,
        num_heads=args.num_heads,
    ).to(device)

    model.load_state_dict(torch.load(pretrain_model))
    model.eval()

    # 根据 model_type 设置 fc 和 sc
    fc = model_type == 'fc' or model_type == 'both'
    sc = model_type == 'sc' or model_type == 'both'
    val_dataset = MyGraphDataset_withPET(table_path,
                                         dmri_graph_dir,
                                         fmri_graph_dir,
                                         PET_dir=pet_dir,
                                         fc_threshold=fc_threshold, sc_threshold=sc_threshold,
                                         data_type='val', fold=fold, ROI=ROI, BOLD=BOLD,pick=pick)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=partial(collate_fn,
                                               mask_ratio=mask_ratio,
                                               fc=fc,
                                               sc=sc))

    with torch.no_grad():

        regression_res_all = {
            'Subject_ids': [],
            'MSE': [],
            'RMSE': [],
            'MAE': [],
            'R2': [],
            'MAPE': []
        }
        for i, batch_data in enumerate(val_loader):
            subject_id = batch_data['subject_ids']
            fc_graphs = batch_data['FC_graphs'].to(device)
            sc_graphs = batch_data['SC_graphs'].to(device)
            pet_label = batch_data['PET_labels'].to(device)

            regression_out,attn_feat,MLP_feat = model(fc_graphs, sc_graphs,args.return_attention_feature,args.return_MLP_feat)

            # -------------------处理regression loss----------
            regression_out = regression_out.squeeze(-1)
            true_values_np = pet_label.cpu().detach().numpy()
            predicted_values_np = regression_out.cpu().detach().numpy()
            for i in range(true_values_np.shape[0]):
                true = true_values_np[i]
                pred = predicted_values_np[i]
                mse, rmse, mae, r2, mape = evaluator_v2(pred, true).values()
                regression_res_all['Subject_ids'].append(subject_id[i])
                regression_res_all['MSE'].append(mse)
                regression_res_all['RMSE'].append(rmse)
                regression_res_all['MAE'].append(mae)
                regression_res_all['R2'].append(r2)
                regression_res_all['MAPE'].append(mape)
                sio.savemat(os.path.join(output_dir, f'{subject_id[i]}_'
                                                     f'MSE{mse:.4f}_'
                                                     f'RMSE{rmse:.4f}_'
                                                     f'MAE{mae:.4f}_'
                                                     f'MAPE{mape:.4f}_'
                                                     f'R2{r2:.4f}.mat'), {
                                'orig': true,
                                'pred': pred,
                                'subid': subject_id[i],
                                'MSE': mse,
                                'RMSE': rmse,
                                'MAE': mae,
                                'MAPE': mape,
                                'R2': r2
                            })
                sio.savemat(os.path.join(output_dir, f'{subject_id[i]}_attn_feat.mat'), {
                    'attn_feat': attn_feat[i].to('cpu').numpy()
                })
                sio.savemat(os.path.join(output_dir, f'{subject_id[i]}_MLP_feat.mat'), {
                    'MLP_feat': MLP_feat[args.return_MLP_feat-1][i].to('cpu').numpy()
                })

        # 将 regression_res_all 转换为 DataFrame
        regression_results_df = pd.DataFrame(regression_res_all)
        # 计算排名，从大到小（越大排名越小）
        regression_results_df['MSE_Rank'] = regression_results_df['MSE'].rank(ascending=False)
        regression_results_df['RMSE_Rank'] = regression_results_df['RMSE'].rank(ascending=False)
        regression_results_df['MAE_Rank'] = regression_results_df['MAE'].rank(ascending=False)
        regression_results_df['R2_Rank'] = regression_results_df['R2'].rank(ascending=False)
        regression_results_df['MAPE_Rank'] = regression_results_df['MAPE'].rank(ascending=False)
        # 选择需要保存的列
        ranking_df = regression_results_df[['Subject_ids', 'MSE_Rank', 'RMSE_Rank', 'MAE_Rank', 'R2_Rank', 'MAPE_Rank']]
        # 保存为 CSV 文件
        ranking_df.to_csv(os.path.join(output_dir, 'subject_ranking.csv'), index=False)

        print('Regression')
        print(args.pretrained)
        print(f'MSE {np.mean(regression_res_all["MSE"]):.4f}_{np.std(regression_res_all["MSE"]):.4f} '
              f'RMSE {np.mean(regression_res_all["RMSE"]):.4f}_{np.std(regression_res_all["RMSE"]):.4f} '
              f'MAE {np.mean(regression_res_all["MAE"]):.4f}_{np.std(regression_res_all["MAE"]):.4f} '
              f'MAPE {np.mean(regression_res_all["MAPE"]):.4f}_{np.std(regression_res_all["MAPE"]):.4f} '
              f'R2 {np.mean(regression_res_all["R2"]):.4f}_{np.std(regression_res_all["R2"]):.4f}')

