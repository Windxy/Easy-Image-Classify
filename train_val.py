# coding:utf-8
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import datasets, transforms, models
import numpy as np
import argparse
from tqdm import tqdm
import sys, time
from tensorboardX import SummaryWriter

from utils.util import load_checkpoint, save_checkpoint, ensure_dir
from model import example_model

# set flags / seeds
torch.backends.cudnn.benchmark = True
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)


class Logger(object):
    def __init__(self, filename='train.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


sys.stdout = Logger('Model.log', sys.stdout)

# Start with main code
if __name__ == '__main__':
    '''step0.参数配置'''
    parser = argparse.ArgumentParser(description="图像分类代码示例")
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='指定优化器的初始学习率 默认1e-3)')
    parser.add_argument('--resume', action='store_true',
                        help='使用前一次保存的权重 默认None，可选True or False')
    parser.add_argument('--path_to_checkpoint', type=str, default='',
                        help='前一次保存权重的地址 默认:""')
    parser.add_argument('--epochs', type=int, default=5,
                        help='epoch数 默认50')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size大小 默认32')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='数据集加载进程数 默认8)')

    opt = parser.parse_args()

    '''step1.设置数据集和对应的transform'''
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='cifar10',
                                     train=True,
                                     transform=train_transforms,
                                     download=True)
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=opt.batch_size,
                                        shuffle=True,
                                        drop_last=True,
                                        num_workers=opt.num_workers)

    test_dataset = datasets.CIFAR10(root='cifar10',
                                    train=False,
                                    transform=test_transforms,
                                    download=True)
    test_data_loader = data.DataLoader(test_dataset,
                                       batch_size=opt.batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers)

    '''step2.加载网络'''
    net = example_model.ExampleModel()

    '''step3.加载损失函数'''
    criterion_CE = nn.CrossEntropyLoss()

    # gpu加速
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net = net.cuda()

    '''step4.加载学习器'''
    optim = torch.optim.Adam(net.parameters(), lr=opt.lr)

    '''step5.加载预训练权重'''
    start_n_iter = 0
    start_epoch = 0
    if opt.resume:
        ckpt = load_checkpoint(opt.path_to_checkpoint)  # custom method for loading last checkpoint
        net.load_state_dict(ckpt['net'])
        start_epoch = ckpt['epoch']
        start_n_iter = ckpt['n_iter']
        optim.load_state_dict(ckpt['optim'])
        print("last checkpoint restored")

    # 用tensorboardX跟踪进展
    writer = SummaryWriter()

    '''step6.开始训练'''
    n_iter = start_n_iter
    for epoch in range(start_epoch, opt.epochs):
        # 设置为train模式
        net.train()

        # 使用tqdm进行迭代
        pbar = tqdm(enumerate(train_data_loader),
                    total=len(train_data_loader))
        start_time = time.time()

        # 数据集批处理
        for i, data in pbar:
            '''step6.1数据加载'''
            img, label = data
            if use_cuda:
                img = img.cuda()
                label = label.cuda()

            # 使用tqdm跟踪准备时间和计算时间
            prepare_time = start_time - time.time()

            '''step6.2前向计算'''
            out = net(img)
            '''step6.3损失计算'''
            loss = criterion_CE(out, label)
            '''step6.4学习器梯度清零'''
            optim.zero_grad()
            '''step6.5后向梯度计算'''
            loss.backward()
            '''step6.6参数更新'''
            optim.step()

            # 更新 tensorboardX
            writer.add_scalar('train_loss', n_iter)

            # 计算时间和计算效率computation time & *compute_efficiency*
            process_time = start_time - time.time() - prepare_time
            compute_efficiency = process_time / (process_time + prepare_time)
            pbar.set_description(
                f'Compute efficiency: {compute_efficiency:.2f}, '
                f'loss: {loss.item():.2f},  epoch: {epoch}/{opt.epochs}')
            start_time = time.time()

        # 验证测试
        if epoch % 1 == 0:
            # 设置为evel模式
            net.eval()

            correct = 0
            total = 0

            pbar = tqdm(enumerate(test_data_loader),
                        total=len(test_data_loader))
            with torch.no_grad():
                for i, data in pbar:
                    # data preparation
                    img, label = data
                    if use_cuda:
                        img = img.cuda()
                        label = label.cuda()

                    out = net(img)
                    _, predicted = torch.max(out.data, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

            print(f'Accuracy on test set: {100 * correct / total:.2f}')

            # 保存历史
            cpkt = {
                'net': net.state_dict(),
                'epoch': epoch,
                'n_iter': n_iter,
                'optim': optim.state_dict()
            }
            save_checkpoint(cpkt, 'model_checkpoint.ckpt')
