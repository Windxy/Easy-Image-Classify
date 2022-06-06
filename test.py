# coding:utf-8
import torch
from torch.utils import data
from torchvision import datasets, transforms
import numpy as np
import argparse
from tqdm import tqdm
import time

from utils.util import load_checkpoint
from model import example_model

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

# Start with main code
if __name__ == '__main__':
    '''step0.参数配置'''
    parser = argparse.ArgumentParser(description="图像分类代码示例")
    parser.add_argument('--path_to_checkpoint', type=str, default='model_checkpoint.ckpt',
                        help='前一次保存权重的地址 默认:""')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据集加载进程数 默认0')
    parser.add_argument('--Dataset', type=str, default='custom', choices=['custom'],
                        help='选择使用测试的数据集')
    parser.add_argument('--input_size', type=int, default=28,
                        help='输入图像的尺寸，例如,28*28')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='类别数，多少个类别就是多少')

    opt = parser.parse_args()

    '''step1.设置数据集和对应的transform'''
    test_transforms = transforms.Compose([
        transforms.RandomCrop(opt.input_size, padding=4),
        # transforms.Resize(opt.input_size),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # 与train_val.py中保持一致
    ])

    assert opt.Dataset == 'custom'
    test_dataset = datasets.ImageFolder(root="CustomDataSet\\test", transform=test_transforms)

    test_data_loader = data.DataLoader(test_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        drop_last=False,
                                        num_workers=opt.num_workers)

    '''step2.加载网络'''
    net = example_model.ExampleModel(class_num=opt.num_classes)

    # gpu加速
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()

    '''step3.加载预训练权重'''
    start_n_iter = 0
    start_epoch = 0
    assert opt.path_to_checkpoint!=''
    ckpt = load_checkpoint(opt.path_to_checkpoint)
    net.load_state_dict(ckpt['net'])
    print("load ckpt successful!")

    '''step4.开始测试'''
    # 设置为eval模式
    net.eval()

    # 使用tqdm进行迭代
    pbar = tqdm(enumerate(test_data_loader),total=len(test_data_loader))
    start_time = time.time()
    # 验证测试
    correct = 0
    total = 0

    y_true = []
    y_score = []
    y_pre = []

    with torch.no_grad():
        for i, data in pbar:
            # 准备数据
            img, label = data
            if use_cuda:
                img = img.cuda()
                label = label.cuda()

            out = net(img)
            score, predicted = torch.max(out.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    print(f'Accuracy on val set: {100 * correct / total:.2f}')