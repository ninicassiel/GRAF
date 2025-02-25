# -*- coding:UTF-8 -*-
'''
@Date   :{DATE}
'''
import sys
import os
project_root = os.path.abspath("D:\Projects\p#001DTI2PET\dmri_fmri2PET")
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

from mymodels.PyG_tools.dataset import MyGraphDataset, collate_fn, MyGraphDataset_withPET

# from models import MaskGAE,MaskGAE_single
from models import MaskGAE_stage1
from mymodels.PyG_models_v8_4.models_2stages import MaskGAE
from torch_geometric.data import Batch
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import scipy.io as sio

from mymodels.PyG_tools.evaluator import evaluator,evaluator_v2,evaluator_v3
from mymodels.PyG_tools.get_loss_fn import get_loss_fn
from mymodels.PyG_tools.save_json import save_args_to_json

def load_pretrain_model(pretrain_model_path):
    config_path = os.path.join(os.path.dirname(pretrain_model_path), 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
        if config['decoder_type']=='mlp':
            pred_model = MaskGAE(
                in_node_channels=config['in_node_channels'],
                in_edge_channels=config['in_edge_channels'],
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
            pred_model = MaskGAE(
                in_node_channels=config['in_node_channels'],
                in_edge_channels=config['in_edge_channels'],
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
    loss_type = args.loss
    loss_weight = args.loss_weight
    model_type = args.model_type
    random_frequency = args.random_frequency

    # 根据 model_type 设置 fc 和 sc
    fc = model_type == 'fc' or model_type == 'both'
    sc = model_type == 'sc' or model_type == 'both'
    decoder_pretrained = args.decoder_pretrained

    if args.minitest:
        train_dataset = MyGraphDataset(table_path,
                                       dmri_graph_dir,
                                       fmri_graph_dir,
                                       fc_threshold=fc_threshold, sc_threshold=sc_threshold,
                                       data_type='val', fold=fold, ROI=ROI, BOLD=BOLD,is_NC_only=args.is_NC_only)
    else:
        train_dataset = MyGraphDataset(table_path,
                                       dmri_graph_dir,
                                       fmri_graph_dir,
                                       fc_threshold = fc_threshold,sc_threshold = sc_threshold,
                                       data_type='train',fold=fold, ROI=ROI, BOLD=BOLD,is_NC_only=args.is_NC_only)
    val_dataset = MyGraphDataset(table_path,
                                 dmri_graph_dir,
                                 fmri_graph_dir,
                                 fc_threshold = fc_threshold,sc_threshold = sc_threshold,
                                 data_type='val',fold=fold, ROI=ROI, BOLD=BOLD)
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
    if args.decoder_type=='mlp':
        model = MaskGAE_stage1(
            encoder_hidden_channels=args.encoder_hidden_channels,
            encoder_node_out_channels=args.encoder_node_out_channels,
            encoder_edge_out_channels=args.encoder_edge_out_channels,
            encoder_num_layers=args.encoder_num_layers,
            encoder_dropout=args.encoder_dropout,
            decoder_type='mlp',
            edge_decoder_num_layers=args.edge_decoder_num_layers,
            edge_decoder_hidden_channels=args.edge_decoder_hidden_channels,
            edge_decoder_dropout=args.edge_decoder_dropout,
            degree_decoder_hidden_channels=args.degree_decoder_hidden_channels,
            degree_decoder_num_layers=args.degree_decoder_num_layers,
            degree_decoder_dropout=args.degree_decoder_dropout,
            deg=args.deg
        ).to(device)
    else:
        model = MaskGAE_stage1(

            encoder_hidden_channels=args.encoder_hidden_channels,
            encoder_node_out_channels=args.encoder_node_out_channels,
            encoder_edge_out_channels=args.encoder_edge_out_channels,
            encoder_num_layers=args.encoder_num_layers,
            encoder_dropout=args.encoder_dropout,
            decoder_type='dot',
            degree_decoder_hidden_channels=args.degree_decoder_hidden_channels,
            degree_decoder_num_layers=args.degree_decoder_num_layers,
            degree_decoder_dropout=args.degree_decoder_dropout,
            deg=args.deg
        ).to(device)
    if decoder_pretrained:
        fc_pretrain_path = args.fc_pretrained
        fc_pred_model = load_pretrain_model(fc_pretrain_path).to(device)
        sc_pretrain_path = args.sc_pretrained
        sc_pred_model = load_pretrain_model(sc_pretrain_path).to(device)

        # print(model.fc_encoder)
        # print(fc_pred_model.encoder)
        # print(model.sc_encoder)
        # print(sc_pred_model.encoder)

        model.sc_encoder.load_state_dict(sc_pred_model.encoder.state_dict())
        model.fc_encoder.load_state_dict(fc_pred_model.encoder.state_dict())
        # if args.decoder_type=='mlp':
        #     model.fc_edge_decoder.load_state_dict(fc_pred_model.edge_decoder.state_dict())
        #     model.sc_edge_decoder.load_state_dict(sc_pred_model.edge_decoder.state_dict())

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)
    criterion = get_loss_fn(loss_type)

    best_val_loss = float('inf')  # Initialize best validation loss
    best_fc_auc = -float('inf')  # Initialize best validation accuracy
    best_sc_auc = -float('inf')  # Initialize best validation auc
    model.train()
    for e in range(epoch):
        if e % random_frequency == 0:
            seed = e // random_frequency
            random.seed(seed)
            torch.manual_seed(seed)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                      collate_fn=partial(collate_fn,
                                                         mask_ratio=mask_ratio,
                                                         fc=fc,
                                                         sc=sc,
                                                         seed = seed))
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    collate_fn=partial(collate_fn,
                                                       mask_ratio=mask_ratio,
                                                       fc=fc,
                                                       sc=sc,
                                                       seed=seed))
        model.train()
        total_loss = []
        fc_edge_loss = []
        sc_edge_loss = []
        if args.deg:
            fc_degree_loss = []
            sc_degree_loss = []
        for i, batch_data in enumerate(train_loader):
            # start_t = time.time()
            optimizer.zero_grad()  # 清除之前的梯度

            fc_graphs = batch_data['FC_graphs'].to(device)
            sc_graphs = batch_data['SC_graphs'].to(device)
            mask_ratio = mask_ratio
            (fc_pos_out,fc_neg_out,fc_deg,fc_deg_out,
             sc_pos_out,sc_neg_out,sc_deg,sc_deg_out)= model(fc_graphs,sc_graphs,sigmoid = False)

            fc_pos_out = fc_pos_out.squeeze(-1)
            fc_neg_out = fc_neg_out.squeeze(-1)

            # fc_pos_weight = fc_neg_out.shape[0] / fc_pos_out.shape[0] if fc_pos_out.shape[0] > 0 else None
            # fc_loss = criterion(fc_pos_out, fc_neg_out, fc_pos_weight if loss_weight else None)
            fc_loss = criterion(fc_pos_out, fc_neg_out, args.loss_weight if loss_weight else None)
            fc_edge_loss.append(fc_loss.item())

            # -------------------处理sc loss
            sc_pos_out = sc_pos_out.squeeze(-1)
            sc_neg_out = sc_neg_out.squeeze(-1)

            # sc_pos_weight = sc_neg_out.shape[0] / sc_pos_out.shape[0] if sc_pos_out.shape[0] > 0 else None
            # sc_loss = criterion(sc_pos_out, sc_neg_out, sc_pos_weight if loss_weight else None)
            sc_loss = criterion(sc_pos_out, sc_neg_out, args.loss_weight if loss_weight else None)
            sc_edge_loss.append(sc_loss.item())

            total_loss.append(fc_edge_loss[-1]+sc_edge_loss[-1])

            # # 不同模态的loss归一化到1
            fc_loss_no_grad = fc_loss.detach()
            sc_loss_no_grad = sc_loss.detach()
            fc_loss_normalized = fc_loss / fc_loss_no_grad
            sc_loss_normalized = sc_loss / sc_loss_no_grad
            loss = fc_loss_normalized+sc_loss_normalized


            loss.backward()
            optimizer.step()
            # end_t = time.time()
            # print(f'backward time: {end_t-start_t}s')
            total_loss.append(loss.item())

        val_total_loss = []
        val_fc_edge_loss = []
        val_sc_edge_loss = []
        model.eval()
        results_all = {key: [] for key in
                       ['AUC_fc', 'Accuracy_fc', 'Precision_fc', 'F1_fc', 'Sensitivity_fc', 'Specificity_fc',
                        'AUC_sc', 'Accuracy_sc', 'Precision_sc', 'F1_sc', 'Sensitivity_sc', 'Specificity_sc']}

        with torch.no_grad():  # 禁用梯度计算，节省内存和加速验证
            for i, batch_data in enumerate(val_loader):
                # start_t = time.time()
                fc_graphs = batch_data['FC_graphs'].to(device)
                sc_graphs = batch_data['SC_graphs'].to(device)
                mask_ratio = mask_ratio
                # start = time.time()
                fc_pos_out, fc_neg_out, fc_deg, fc_deg_out, sc_pos_out, sc_neg_out, sc_deg, sc_deg_out = model(
                    fc_graphs,
                    sc_graphs,
                    sigmoid = False)

                #  ----------------处理fc------------------------
                fc_pos_out = fc_pos_out.squeeze(-1)
                fc_neg_out = fc_neg_out.squeeze(-1)

                fc_loss = criterion(fc_pos_out, fc_neg_out, args.loss_weight if loss_weight else None)
                val_fc_edge_loss.append(fc_loss.item())

                #  ----------------处理sc------------------------
                sc_pos_out = sc_pos_out.squeeze(-1)
                sc_neg_out = sc_neg_out.squeeze(-1)

                # sc_pos_weight = sc_neg_out.shape[0] / sc_pos_out.shape[0] if sc_pos_out.shape[0] > 0 else None
                sc_loss = criterion(sc_pos_out, sc_neg_out, args.loss_weight if loss_weight else None)
                val_sc_edge_loss.append(sc_loss.item())
                val_sc_edge_loss.append(sc_loss.item())

                val_total_loss.append(val_fc_edge_loss[-1]+val_sc_edge_loss[-1])
                fc_results = evaluator(fc_pos_out.detach().cpu(), fc_neg_out.detach().cpu())
                sc_results = evaluator(sc_pos_out.detach().cpu(), sc_neg_out.detach().cpu())

                # 更新结果, 避免循环中进行多次 if 检查
                for key, value in fc_results.items():
                    results_all[key + '_fc'].append(value)
                # 更新结果, 避免循环中进行多次 if 检查
                for key, value in sc_results.items():
                    results_all[key + '_sc'].append(value)

        # 计算所有损失和指标的均值
        train_losses = {
            'loss': np.mean(total_loss),
            'fc_edge_loss': np.mean(fc_edge_loss),
            'sc_edge_loss': np.mean(sc_edge_loss),
        }

        val_losses = {
            'loss': np.mean(val_total_loss),
            'fc_edge_loss': np.mean(val_fc_edge_loss),
            'sc_edge_loss': np.mean(val_sc_edge_loss),
        }

        results = {
            'AUC_fc': np.mean(results_all['AUC_fc']),
            'AUC_sc': np.mean(results_all['AUC_sc']),
            'Accuracy_fc': np.mean(results_all['Accuracy_fc']),
            'Accuracy_sc': np.mean(results_all['Accuracy_sc']),
            'Precision_fc': np.mean(results_all['Precision_fc']),
            'Precision_sc': np.mean(results_all['Precision_sc']),
            'F1_fc': np.mean(results_all['F1_fc']),
            'F1_sc': np.mean(results_all['F1_sc']),
            'Sensitivity_fc': np.mean(results_all['Sensitivity_fc']),
            'Sensitivity_sc': np.mean(results_all['Sensitivity_sc']),
            'Specificity_fc': np.mean(results_all['Specificity_fc']),
            'Specificity_sc': np.mean(results_all['Specificity_sc']),
        }

        # 记录到 TensorBoard
        writer.add_scalars('loss', {'train': train_losses['loss'], 'val': val_losses['loss']}, e)
        writer.add_scalars('loss', {'train': train_losses['loss']}, e)
        writer.add_scalars('fc_edge_loss', {'train': train_losses['fc_edge_loss'], 'val': val_losses['fc_edge_loss']},
                           e)
        writer.add_scalars('sc_edge_loss', {'train': train_losses['sc_edge_loss'], 'val': val_losses['sc_edge_loss']},
                           e)
        writer.add_scalars('AUC', {'fc': results['AUC_fc'], 'sc': results['AUC_sc']}, e)
        writer.add_scalars('Accuracy', {'fc': results['Accuracy_fc'], 'sc': results['Accuracy_sc']}, e)
        writer.add_scalars('Precision', {'fc': results['Precision_fc'], 'sc': results['Precision_sc']}, e)
        writer.add_scalars('F1', {'fc': results['F1_fc'], 'sc': results['F1_sc']}, e)
        writer.add_scalars('Sensitivity', {'fc': results['Sensitivity_fc'], 'sc': results['Sensitivity_sc']}, e)
        writer.add_scalars('Specificity', {'fc': results['Specificity_fc'], 'sc': results['Specificity_sc']}, e)

        # 更新最优模型
        if (val_losses['loss'] < best_val_loss or
                best_fc_auc > results['AUC_fc'] or best_sc_auc > results['AUC_sc']):
            # 更新最佳验证损失和准确率
            best_val_loss = min(best_val_loss, val_losses['loss'])
            best_fc_auc = max(best_fc_auc, results['AUC_fc'])
            best_sc_auc = max(best_sc_auc, results['AUC_sc'])

            # 构建模型路径
            model_path = os.path.join(output_dir,
                                      f"Best_model_epoch_{e:03d}_loss{val_losses['loss']:.4f}_fcAcc{results['Accuracy_fc']:.4f}_fcAuc{results['AUC_fc']:.4f}_scAcc{results['Accuracy_sc']:.4f}_scAuc{results['AUC_sc']:.4f}.pth")

            # 保存模型
            torch.save(model.state_dict(), model_path)

        print(f'Epoch [{e:03d}] train loss: {np.mean(total_loss):.4f} val loss: {np.mean(val_total_loss):.4f} ')
        # print(f'Epoch [{e:03d}]')
        print(
            f"FC: AUC: {results['AUC_fc']:.4f}, Accuracy: {results['Accuracy_fc']:.4f}, Sensitivity: {results['Sensitivity_fc']:.4f}, Specificity: {results['Specificity_fc']:.4f}")
        print(
            f"SC: AUC: {results['AUC_sc']:.4f}, Accuracy: {results['Accuracy_sc']:.4f}, Sensitivity: {results['Sensitivity_sc']:.4f}, Specificity: {results['Specificity_sc']:.4f}")
    writer.close()

def test_visualize(fold,args):
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

    pretrain_model = args.pretrained
    epoch = os.path.split(pretrain_model)[-1].split('_')[3]
    output_dir = f'{os.path.dirname(pretrain_model)}/validation_epoch{epoch}'
    os.makedirs(output_dir, exist_ok=True)

    ROI = args.ROI
    BOLD = args.BOLD
    batch_size = args.batch_size
    fc_threshold = args.fc_threshold
    sc_threshold = args.sc_threshold
    mask_ratio = args.mask_ratio
    model_type = args.model_type

    if args.decoder_type == 'mlp':
        model = MaskGAE_stage1(
            encoder_hidden_channels=args.encoder_hidden_channels,
            encoder_node_out_channels=args.encoder_node_out_channels,
            encoder_edge_out_channels=args.encoder_edge_out_channels,
            encoder_num_layers=args.encoder_num_layers,
            encoder_dropout=args.encoder_dropout,
            decoder_type='mlp',
            edge_decoder_num_layers=args.edge_decoder_num_layers,
            edge_decoder_hidden_channels=args.edge_decoder_hidden_channels,
            edge_decoder_dropout=args.edge_decoder_dropout,
            degree_decoder_hidden_channels=args.degree_decoder_hidden_channels,
            degree_decoder_num_layers=args.degree_decoder_num_layers,
            degree_decoder_dropout=args.degree_decoder_dropout,
            deg=args.deg
        ).to(device)
    else:
        model = MaskGAE_stage1(

            encoder_hidden_channels=args.encoder_hidden_channels,
            encoder_node_out_channels=args.encoder_node_out_channels,
            encoder_edge_out_channels=args.encoder_edge_out_channels,
            encoder_num_layers=args.encoder_num_layers,
            encoder_dropout=args.encoder_dropout,
            decoder_type='dot',
            degree_decoder_hidden_channels=args.degree_decoder_hidden_channels,
            degree_decoder_num_layers=args.degree_decoder_num_layers,
            degree_decoder_dropout=args.degree_decoder_dropout,
            deg=args.deg
        ).to(device)

    model.load_state_dict(torch.load(pretrain_model))
    model.eval()

    # 根据 model_type 设置 fc 和 sc
    fc = model_type == 'fc' or model_type == 'both'
    sc = model_type == 'sc' or model_type == 'both'
    val_dataset = MyGraphDataset(table_path,
                                 dmri_graph_dir,
                                 fmri_graph_dir,
                                 fc_threshold=fc_threshold, sc_threshold=sc_threshold,
                                 data_type='val', fold=fold, ROI=ROI, BOLD=BOLD)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=partial(collate_fn,
                                               mask_ratio=mask_ratio,
                                               fc=fc,
                                               sc=sc))

    fc_output_dir = os.path.join(output_dir, 'fc')
    os.makedirs(fc_output_dir, exist_ok=True)
    sc_output_dir = os.path.join(output_dir, 'sc')
    os.makedirs(sc_output_dir, exist_ok=True)
    fc_res_all = {
        'Subject_ids': [],
        'Accuracy': [],
        'AUC': [],
        'Sensitivity': [],
        'Specificity': []
    }
    sc_res_all = {
        'Subject_ids': [],
        'Accuracy': [],
        'AUC': [],
        'Sensitivity': [],
        'Specificity': []
    }

    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            subject_id = batch_data['subject_ids']
            fc_graphs = batch_data['FC_graphs'].to(device)
            sc_graphs = batch_data['SC_graphs'].to(device)
            mask_ratio = mask_ratio
            # start = time.time()
            (fc_pos_out, fc_neg_out, fc_deg, fc_deg_out,
             sc_pos_out, sc_neg_out, sc_deg, sc_deg_out) = model(fc_graphs, sc_graphs)

            #  ----------------处理fc------------------------
            fc_pos_out = fc_pos_out.squeeze(-1)
            fc_neg_out = fc_neg_out.squeeze(-1)

            fc_graphs = fc_graphs.to('cpu')
            pr_mask, pr_neg = 0, 0
            graphs_list = Batch.to_data_list(fc_graphs)
            fc_pos_out = fc_pos_out.to('cpu')
            fc_neg_out = fc_neg_out.to('cpu')
            for j, graph in enumerate(graphs_list):
                sid = subject_id[j]
                mask_edge_index_sample = graph.edge_index[:, graph.edge_mask == 1]
                remain_edge_index_sample = graph.edge_index[:, graph.edge_mask == 0]
                neg_edge_index_sample = graph.edge_index[:, graph.neg_mask == 1]
                mask_edge_num = mask_edge_index_sample.shape[1]
                neg_edge_num = neg_edge_index_sample.shape[1]
                pos_out_sample = fc_pos_out[pr_mask:pr_mask + mask_edge_num].squeeze(-1)
                neg_out_sample = fc_neg_out[pr_neg:pr_neg + neg_edge_num].squeeze(-1)
                pr_mask += mask_edge_num
                pr_neg += neg_edge_num
                res = {
                    'remain_edge': remain_edge_index_sample.cpu().numpy(),
                    'mask_edge': mask_edge_index_sample.cpu().numpy(),
                    'neg_edge': neg_edge_index_sample.cpu().numpy(),
                    'pos_out': pos_out_sample.cpu().numpy(),
                    'neg_out': neg_out_sample.cpu().numpy()
                }
                eres = evaluator(pos_out_sample, neg_out_sample)
                for key in eres.keys():
                    res[key] = eres[key]
                sio.savemat(os.path.join(fc_output_dir,
                                         f'{sid}_Acc{res["Accuracy"]:.4f}_AUC{res["AUC"]:.4f}_Sensitivity{res["Sensitivity"]:.4f}_Specificity{res["Specificity"]:.4f}.mat'),
                            res)
                fc_res_all['Subject_ids'].append(sid)
                fc_res_all['Accuracy'].append(res['Accuracy'])
                fc_res_all['AUC'].append(res['AUC'])
                fc_res_all['Sensitivity'].append(res['Sensitivity'])
                fc_res_all['Specificity'].append(res['Specificity'])
            # ----------------处理sc------------------------
            sc_pos_out = sc_pos_out.squeeze(-1)
            sc_neg_out = sc_neg_out.squeeze(-1)

            sc_graphs = sc_graphs.to('cpu')
            pr_mask, pr_neg = 0, 0
            graphs_list = Batch.to_data_list(sc_graphs)
            sc_pos_out = sc_pos_out.to('cpu')
            sc_neg_out = sc_neg_out.to('cpu')
            for j, graph in enumerate(graphs_list):
                sid = subject_id[j]
                mask_edge_index_sample = graph.edge_index[:, graph.edge_mask == 1]
                remain_edge_index_sample = graph.edge_index[:, graph.edge_mask == 0]
                neg_edge_index_sample = graph.edge_index[:, graph.neg_mask == 1]
                mask_edge_num = mask_edge_index_sample.shape[1]
                neg_edge_num = neg_edge_index_sample.shape[1]
                pos_out_sample = sc_pos_out[pr_mask:pr_mask + mask_edge_num].squeeze(-1)
                neg_out_sample = sc_neg_out[pr_neg:pr_neg + neg_edge_num].squeeze(-1)
                pr_mask += mask_edge_num
                pr_neg += neg_edge_num
                res = {
                    'remain_edge': remain_edge_index_sample.cpu().numpy(),
                    'mask_edge': mask_edge_index_sample.cpu().numpy(),
                    'neg_edge': neg_edge_index_sample.cpu().numpy(),
                    'pos_out': pos_out_sample.cpu().numpy(),
                    'neg_out': neg_out_sample.cpu().numpy()
                }
                eres = evaluator(pos_out_sample, neg_out_sample)
                for key in eres.keys():
                    res[key] = eres[key]
                sio.savemat(os.path.join(sc_output_dir,
                                         f'{sid}_Acc{res["Accuracy"]:.4f}_AUC{res["AUC"]:.4f}_Sensitivity{res["Sensitivity"]:.4f}_Specificity{res["Specificity"]:.4f}.mat'),
                            res)
                sc_res_all['Subject_ids'].append(sid)
                sc_res_all['Accuracy'].append(res['Accuracy'])
                sc_res_all['AUC'].append(res['AUC'])
                sc_res_all['Sensitivity'].append(res['Sensitivity'])
                sc_res_all['Specificity'].append(res['Specificity'])

        print('-------------------fc-----------------------')
        print(f'Accuracy: {np.mean(fc_res_all["Accuracy"]):.4f}_{np.std(fc_res_all["Accuracy"]):.4f}')
        print(f'AUC: {np.mean(fc_res_all["AUC"]):.4f}_{np.std(fc_res_all["AUC"]):.4f}')
        print(f'Sensitivity: {np.mean(fc_res_all["Sensitivity"]):.4f}_{np.std(fc_res_all["Sensitivity"]):.4f}')
        print(f'Specificity: {np.mean(fc_res_all["Specificity"]):.4f}_{np.std(fc_res_all["Specificity"]):.4f}')

        import pandas as pd
        # 将 fc_res_all 转换为 DataFrame
        fc_results_df = pd.DataFrame(fc_res_all)

        # 计算排名，从大到小（越大排名越小）
        fc_results_df['ACC_Rank'] = fc_results_df['Accuracy'].rank(ascending=False)
        fc_results_df['AUC_Rank'] = fc_results_df['AUC'].rank(ascending=False)
        fc_results_df['Sensitivity_Rank'] = fc_results_df['Sensitivity'].rank(ascending=False)
        fc_results_df['Specificity_Rank'] = fc_results_df['Specificity'].rank(ascending=False)

        # 选择需要保存的列
        ranking_df = fc_results_df[['Subject_ids', 'ACC_Rank', 'AUC_Rank', 'Sensitivity_Rank', 'Specificity_Rank']]

        # 保存为 CSV 文件
        ranking_df.to_csv(os.path.join(fc_output_dir, 'subject_ranking.csv'), index=False)

        print('-------------------sc-----------------------')
        print(f'Accuracy: {np.mean(sc_res_all["Accuracy"]):.4f}_{np.std(sc_res_all["Accuracy"]):.4f}')
        print(f'AUC: {np.mean(sc_res_all["AUC"]):.4f}_{np.std(sc_res_all["AUC"]):.4f}')
        print(f'Sensitivity: {np.mean(sc_res_all["Sensitivity"]):.4f}_{np.std(sc_res_all["Sensitivity"]):.4f}')
        print(f'Specificity: {np.mean(sc_res_all["Specificity"]):.4f}_{np.std(sc_res_all["Specificity"]):.4f}')

        import pandas as pd
        # 将 sc_res_all 转换为 DataFrame
        sc_results_df = pd.DataFrame(sc_res_all)

        # 计算排名，从大到小（越大排名越小）
        sc_results_df['ACC_Rank'] = sc_results_df['Accuracy'].rank(ascending=False)
        sc_results_df['AUC_Rank'] = sc_results_df['AUC'].rank(ascending=False)
        sc_results_df['Sensitivity_Rank'] = sc_results_df['Sensitivity'].rank(ascending=False)
        sc_results_df['Specificity_Rank'] = sc_results_df['Specificity'].rank(ascending=False)

        # 选择需要保存的列
        ranking_df = sc_results_df[['Subject_ids', 'ACC_Rank', 'AUC_Rank', 'Sensitivity_Rank', 'Specificity_Rank']]

        # 保存为 CSV 文件
        ranking_df.to_csv(os.path.join(sc_output_dir, 'subject_ranking.csv'), index=False)

        os.rename(output_dir, f'{output_dir}_fcAcc{np.mean(fc_res_all["Accuracy"]):.4f}_fcAuc{np.mean(fc_res_all["AUC"]):.4f}_scAcc{np.mean(sc_res_all["Accuracy"]):.4f}_scAuc{np.mean(sc_res_all["AUC"]):.4f}')
