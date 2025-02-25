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
    parser.add_argument('--base_dir', default='/GRAFS/data', help='Base directory path.')
    parser.add_argument('--table_path_file',
                        default='SubjectsTable.csv',
                        help='Path to table file.')
    parser.add_argument('--dmri_graph_dir', type=str, default="dmri_sc", help='Directory for DMRi graphs.')
    parser.add_argument('--fmri_graph_dir', type=str, default="fmri_fc", help='Directory for fMRI graphs.')
    parser.add_argument('--pet_dir', type=str, default="PET_suvr", help='Directory for PET data.')

    parser.add_argument('--num_folds', type=int, default=1, help='Number of folds.')
    parser.add_argument('--model_type', default='sc', help='Model type.')
    parser.add_argument('--ROI', type=int, default=100, help='Number of ROIs.')
    parser.add_argument('--BOLD', action='store_true', help='BOLD flag.')
    parser.add_argument('--fc_threshold', type=float, default=0.4, help='FC threshold.')
    parser.add_argument('--sc_threshold', type=float, default=0.4, help='SC threshold.')
    parser.add_argument('--mask_ratio', type=float, default=0.2, help='Mask ratio.')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--alpha', type=float, default=0.01, help='Alpha value.')
    parser.add_argument('--loss', default='l1', help='Loss function. bce,l1,acc')
    parser.add_argument('--decoder_type',type=str,default='mlp',help='Decoder type: mlp,dot')

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
    parser.add_argument('--cross_fusion_type', type=str, default='cat', help='Cross fussion type: cat,attn')

    parser.add_argument('--return_attention_feature',type = bool,default=True,help='Return attention feature flag.')
    parser.add_argument('--return_MLP_feat',type = int,default=2,help='Return MLP feature flag,0 means No,')
    parser.add_argument('--pretrained',
                        default=r'xxx',
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

    # test_visualize(args.num_folds, args,pick=False)
    test_visualize(args.num_folds, args,pick=True)
    # for fold in range(1, num_folds + 1):
    #     test_visualize(fold, output_dir=out_dir, config=config)

