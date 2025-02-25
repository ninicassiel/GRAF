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
    parser.add_argument('--out_dir', type=str, default="D:/Projects/Output/dmri_fmri2PET/PyG_models_v17_1",
                        help='Output directory path.')
    parser.add_argument('--minitest', type=bool, default=False, help='Boolean flag for minitest.')
    parser.add_argument('--table_path_file', type=str,
                        default="T#019fmriROISignalRecoder_woHeadPoor_100_withAbeta_dmri_resplit_fold.csv",
                        help='Path to table file.')
    parser.add_argument('--dmri_graph_dir', type=str, default="014_dmri_sc", help='Directory for DMRi graphs.')
    parser.add_argument('--fmri_graph_dir', type=str, default="011_fmri_fc", help='Directory for fMRI graphs.')

    parser.add_argument('--num_folds', type=int, default=5, help='Number of folds.')
    parser.add_argument('--model_type', type=str, default="both", help='Model type.')
    parser.add_argument('--ROI', type=int, default=100, help='Number of ROIs.')
    parser.add_argument('--BOLD', action='store_true', default=False, help='Boolean flag for BOLD.')
    parser.add_argument('--fc_threshold', type=float, default=0.4, help='FC threshold.')
    parser.add_argument('--sc_threshold', type=float, default=0.4, help='SC threshold.')
    parser.add_argument('--mask_ratio', type=float, default=0.2, help='Mask ratio.')
    parser.add_argument('--is_NC_only', type=bool, default=True, help='Boolean flag for NC only.')

    parser.add_argument('--lr', type=float, default= 0.0005, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay value.')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha value.')
    parser.add_argument('--epoch', type=int, default=2000, help='Number of epochs.')
    parser.add_argument('--random_frequency', type=int, default=100, help='Random frequency.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--loss', type=str, default="bce", help='Loss function."bce","focal" or "l1"')
    parser.add_argument('--loss_weight', type=float, default=1, help='positive loss weight.')
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

    parser.add_argument('--deg', action='store_true', default=False, help='Boolean flag for degree information.')
    parser.add_argument('--degree_decoder_hidden_channels', nargs='+', default=[16,16],
                        help='Number of hidden channels in the degree decoder.')
    parser.add_argument('--degree_decoder_num_layers', type=int, default=3,
                        help='Number of layers in the degree decoder.')
    parser.add_argument('--degree_decoder_dropout', type=float, default=0,
                        help='Dropout rate for the degree decoder.')
    parser.add_argument('--decoder_pretrained', type=bool, default=False, help='Boolean flag for decoder pretrained.')

    parser.add_argument('--fc_pretrained', type=str, default=r'D:\Projects\Output\dmri_fmri2PET\PyG_models_v8_4\2024_12_09_20_57_46_fc_0.4_0.2\fold_5\Best_model_epoch_1836_acc0.7830_loss0.4409_auc0.8729_sens0.7765_spec0.7896.pth', help='Pretrained model path.')
    parser.add_argument('--sc_pretrained', type=str, default=r'D:\Projects\Output\dmri_fmri2PET\PyG_models_v8_4\2024_12_09_20_54_31_sc_0.4_0.2\fold_5\Best_model_epoch_1848_acc0.8399_loss0.3569_auc0.9213_sens0.8433_spec0.8366.pth', help='Pretrained model path.')

    parser.add_argument('--description', type=str, default="sc reconstruction.", help='Description of the experiment.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_command_line_args()
    # 创建输出目录，添加时间戳
    if args.minitest:
        out_dir = os.path.join(args.out_dir, 'test')
    else:
        if args.model_type == 'both':
            out_dir = os.path.join(args.out_dir,
                                   f'{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}_{args.fc_threshold}_{args.sc_threshold}_{args.mask_ratio}')
    os.makedirs(out_dir, exist_ok=True)
    train_validate(args.num_folds, out_dir, args)

