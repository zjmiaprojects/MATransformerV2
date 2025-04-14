

import shutil
import os


def batch_copy_files(source_dir, target_dir):
    # 检查源目录和目标目录是否存在，如果不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # 遍历源目录下的文件
    for file_name in os.listdir(source_dir):
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, 'train_'+file_name)
        # 复制文件
        shutil.copy(source_file, target_file)



source_dir='../em_data/train/lab_512'
target_dir='../em_data/train_cross1/lab_512'
batch_copy_files(source_dir,target_dir)
