# -*- coding:UTF-8 -*-
'''
@Date   :{DATE}
'''
import json
def save_args_to_json(args, config_path):
    args_dict = vars(args)
    with open(config_path, 'w') as f:
        json.dump(args_dict, f, indent=4)