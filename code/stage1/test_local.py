# _*_ coding:UTF_8 _*_
'''
@Date   :{DATE}
'''

import warnings
warnings.filterwarnings("ignore")
from main_fn import train_validate,test_visualize
import os
import json
import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Process some configurations.')
    parser.add_argument('--base_dir', default='D:/yhy/data/huashan', help='Base directory path.')
    parser.add_argument('--table_path_file',
                        default='T#019fmriROISignalRecoder_woHeadPoor_100_withAbeta_dmri_resplit_fold.csv',
                        help='Path to table file.')
    parser.add_argument('--dmri_graph_dir', type=str, default="014_dmri_sc", help='Directory for DMRi graphs.')
    parser.add_argument('--fmri_graph_dir', type=str, default="011_fmri_fc", help='Directory for fMRI graphs.')

    parser.add_argument('--num_folds', type=int, default=2, help='Number of folds.')
    parser.add_argument('--model_type', default='sc', help='Model type.')
    parser.add_argument('--ROI', type=int, default=100, help='Number of ROIs.')
    parser.add_argument('--BOLD', action='store_true', help='BOLD flag.')
    parser.add_argument('--fc_threshold', type=float, default=0.4, help='FC threshold.')
    parser.add_argument('--sc_threshold', type=float, default=0.4, help='SC threshold.')
    parser.add_argument('--mask_ratio', type=float, default=0.2, help='Mask ratio.')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha value.')
    parser.add_argument('--loss', default='l1', help='Loss function. bce,l1,acc')
    parser.add_argument('--loss_weight', action='store_true', help='Loss weight flag.')
    parser.add_argument('--decoder_type',type=str,default='mlp',help='Decoder type: mlp,dot')

    # model parameters
    parser.add_argument('--in_node_channels', type=int, default=3, help='Number of input node channels.')
    parser.add_argument('--in_edge_channels', type=int, default=3, help='Number of input edge channels.')
    parser.add_argument('--encoder_hidden_channels', nargs='+', default=[16, 32],
                        help='Number of hidden channels in the encoder.')
    parser.add_argument('--encoder_node_out_channels', type=int, default=64,
                        help='Number of output node channels in the encoder.')
    parser.add_argument('--encoder_edge_out_channels', type=int, default=64,
                        help='Number of output edge channels in the encoder.')
    parser.add_argument('--encoder_num_layers', type=int, default=3, help='Number of layers in the encoder.')
    parser.add_argument('--encoder_dropout', type=float, default=0, help='Dropout rate for the encoder.')
    parser.add_argument('--edge_decoder_hidden_channels', nargs='+', default=[16],
                        help='Number of hidden channels in the edge decoder.')
    parser.add_argument('--edge_decoder_num_layers', type=int, default=2, help='Number of layers in the edge decoder.')
    parser.add_argument('--edge_decoder_dropout', type=float, default=0, help='Dropout rate for the edge decoder.')
    parser.add_argument('--degree_decoder_hidden_channels', nargs='+', default=[16],
                        help='Number of hidden channels in the degree decoder.')
    parser.add_argument('--degree_decoder_num_layers', type=int, default=2,
                        help='Number of layers in the degree decoder.')
    parser.add_argument('--degree_decoder_dropout', type=float, default=0,
                        help='Dropout rate for the degree decoder.')
    parser.add_argument('--deg', action='store_true', default=False, help='Boolean flag for degree information.')

    parser.add_argument('--pretrained',
                        default=r'D:\Projects\Output\dmri_fmri2PET\PyG_models_v17_1\2024_12_17_20_59_13_0.4_0.4_0.2\fold_1\Best_model_epoch_1843_loss0.9246_fcAcc0.7742_fcAuc0.8600_scAcc0.7717_scAuc0.8604.pth',
                        help='Path to pretrained model.')
    parser.add_argument('--description', default='sc/fc reconstruction.', help='Description of the process.')
    return parser.parse_args()
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

if __name__ == '__main__':
    args = parse_args()

    config = load_config(os.path.join(os.path.dirname(args.pretrained),'config.json'))
    for key, value in vars(args).items():
        if key in ['description','pretrained','base_dir']:
            continue
        if key in config:
            setattr(args, key, config[key])
    # for key, value in vars(args).items():
    #     print(f"{key}: {value}")

    test_visualize(args.num_folds, args)
    # for fold in range(1, num_folds + 1):
    #     test_visualize(fold, output_dir=out_dir, config=config)

