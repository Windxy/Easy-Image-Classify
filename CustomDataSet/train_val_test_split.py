# coding:utf-8
# 将自定义数据集，按照自定义比例随机分为训练验证和测试三个部分

import os
import random
import shutil

root = 'flowers'
# 训练、验证和测试占比0.7、0.1、0.2
train_pro = 0.7
val_pro   = 0.1
test_pro  = 0.2

'''
本路径下的文件树设置为这样子(file tree, an example)：
CustomDataset
│  train_val_test_split.py
├─ flowers
│  ├─daisy
│  │      xx.jpg
│  │      xx.jpg
│  │      ...
│  ├─rose
│  │      xx.jpg
│  │      xx.jpg
│  │      ...
│  ├─tulip
│  │      xx.jpg
│  │      xx.jpg
│  │ 	    ... 
│  │  ...      
'''

def get_dirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)

def train_val_test_split(file_list, train_p = 0.7, val_p = 0.1, test_p = 0.2):
    # 随机采样
    trains = random.sample(file_list, int(len(file_list) * train_p))
    # 剩下中继续采样val & test
    val_test = list(set(file_list) - set(trains))
    vals = random.sample(val_test, int(len(val_test) * (val_p/(val_p+test_p) )))
    tests = list(set(val_test) - set(vals))

    return trains, vals, tests

if __name__ == '__main__':
    data_list = os.listdir(root)

    '''创建新目录'''
    train_val_test = ['train','val','test']
    for tvt in train_val_test:
        for class_name in data_list:
            get_dirs(os.path.join(tvt,class_name))

    for class_name in data_list:
        class_name_dir = os.path.join(root,class_name)
        file_list = os.listdir(class_name_dir)
        # 按比例采样
        trains, vals, tests = train_val_test_split(set(file_list), train_p = 0.7, val_p = 0.1, test_p = 0.2)

        # 训练集
        for data in trains:
            src = os.path.join(root,class_name,data)
            dst = os.path.join('train',class_name,data)
            shutil.copy(src,dst)

        # 验证集
        for data in vals:
            src = os.path.join(root,class_name,data)
            dst = os.path.join('val',class_name,data)
            shutil.copy(src,dst)

        # 测试集
        for data in tests:
            src = os.path.join(root,class_name,data)
            dst = os.path.join('test',class_name,data)
            shutil.copy(src,dst)

    print("Finish")