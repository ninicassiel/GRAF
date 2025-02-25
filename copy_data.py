# -*- coding:UTF-8 -*-
'''
@Date   :{DATE}
'''
import pandas as pd
import os
import shutil
table = 'D:\Projects\GRAFS\data\SubjectsTable.csv'
# df = pd.read_csv(table)
subjects = ['AFM0006', 'AFM0044', 'AFM0046', 'AFM0079', 'AFM0080', 'AFM0139', 'AFM0140', 'AFM0141', 'AFM0142', 'AFM0143', 'AFM0145', 'AFM0147', 'AFM0148', 'AFM0150']
src_dir = r'D:\yhy\data\huashan\013_PET_suvr_ROI100_pt2'
tar_dir = r'D:\Projects\GRAFS\data\PET_SUVR'

for file in os.listdir(src_dir):
    if file.split('.pt')[0] in subjects:
        os.makedirs(tar_dir, exist_ok=True)
        shutil.copy(os.path.join(src_dir, file), os.path.join(tar_dir, file))