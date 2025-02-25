# _*_ coding:UTF_8 _*_
'''
@Date   :{DATE}
'''

import warnings
warnings.filterwarnings("ignore")
from main_fn import train_validate
import os
import json
import argparse
import datetime

def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default="D:/yhy/data/huashan",
                        help='Base directory path.')
    parser.add_argument('--out_dir', type=str, default="D:/Projects/Output/dmri_fmri2PET/PyG_models_v17_2",
                        help='Output directory path.')
    parser.add_argument('--minitest', type=bool, default=False, help='Boolean flag for minitest.')
    parser.add_argument('--table_path_file', type=str,
                        default="T#019fmriROISignalRecoder_woHeadPoor_100_withAbeta_dmri_resplit_inPET_fold.csv",
                        help='Path to table file.')
    parser.add_argument('--dmri_graph_dir', type=str, default="014_dmri_sc", help='Directory for DMRi graphs.')
    parser.add_argument('--fmri_graph_dir', type=str, default="011_fmri_fc", help='Directory for fMRI graphs.')
    parser.add_argument('--pet_dir', type=str, default="013_PET_suvr_ROI100_pt2", help='Directory for PET data.')

    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds.')
    parser.add_argument('--model_type', type=str, default="both", help='Model type.')
    parser.add_argument('--ROI', type=int, default=100, help='Number of ROIs.')
    parser.add_argument('--BOLD', action='store_true', default=False, help='Boolean flag for BOLD.')
    parser.add_argument('--fc_threshold', type=float, default=0.4, help='FC threshold.')
    parser.add_argument('--sc_threshold', type=float, default=0.4, help='SC threshold.')
    parser.add_argument('--mask_ratio', type=float, default=0, help='Mask ratio.')

    parser.add_argument('--lr', type=float, default= 0.0005, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay value.')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha value.')
    parser.add_argument('--epoch', type=int, default=1000, help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')

    parser.add_argument('--encoder_num_layers', type=int, default=3, help='Number of layers in the encoder.')
    parser.add_argument('--encoder_hidden_channels', nargs='+', default=[16, 32],
                        help='Number of hidden channels in the encoder.')
    parser.add_argument('--encoder_node_out_channels', type=int, default=64,
                        help='Number of output node channels in the encoder.')
    parser.add_argument('--encoder_edge_out_channels', type=int, default=64,
                        help='Number of output edge channels in the encoder.')
    parser.add_argument('--encoder_dropout', type=float, default=0, help='Dropout rate for the encoder.')

    parser.add_argument('--abeta_decoder_hidden_channels', nargs='+', default=[1028, 100, 100],
                        help='Number of hidden channels in the abeta decoder.')
    parser.add_argument('--abeta_decoder_num_layers', type=int, default=4,
                        help='Number of layers in the abeta decoder.')
    parser.add_argument('--abeta_decoder_dropout', type=float, default=0,
                        help='Dropout rate for the degree decoder.')
    parser.add_argument('--num_heads', type=int, default=1, help='Number of heads in the cross-attention layer.')

    parser.add_argument('--freeze_encoder',type=bool,default=True,help='Boolean flag for freezing the encoder.')
    parser.add_argument('--cross_fusion_type', type=str, default='attn', help='Cross fussion type: cat,attn')
    parser.add_argument('--pretrained', type=str, default=r'D:\\Projects\\Output\\dmri_fmri2PET\\PyG_models_v17_1_2\\2024_12_24_21_01_44_0.4_0.4_0.2\\fold_1\\Best_model_epoch_1311_loss0.8556_fcAcc0.7912_fcAuc0.8801_scAcc0.7923_scAuc0.8823.pth', help='Pretrained model path.')
    parser.add_argument('--description', type=str, default="Abeta prediction.", help='Description of the experiment.')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed.')
    return parser.parse_args()

if __name__ == '__main__':
    import torch
    import numpy as np
    args = parse_command_line_args()
    # 创建输出目录，添加时间戳
    if args.minitest:
        out_dir = os.path.join(args.out_dir, 'test')
    else:
        out_dir = os.path.join(args.out_dir, f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_{args.model_type}_{args.fc_threshold}_{args.sc_threshold}')
    # os.makedirs(out_dir, exist_ok=True)
    # random_seeds = [42, 114514, 3407, 1334,12345,1234]
    # for seed in random_seeds:
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    out_dir+=f'seed{args.random_seed}'
    train_validate(args.num_folds, out_dir, args)

