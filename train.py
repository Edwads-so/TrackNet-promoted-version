import os
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import Shuttlecock_Trajectory_Dataset
from test import eval_tracknet, eval_inpaintnet
from utils.general import ResumeArgumentParser, get_model, to_img_format
from utils.metric import WBCELoss
from utils.visualize import plot_heatmap_pred_sample, plot_traj_pred_sample, write_to_tb
from refined_model import RefinedTrackNet

def mixup(x, y, alpha=0.5):
    batch_size = x.size()[0]
    lamb = np.random.beta(alpha, alpha, size=batch_size)
    lamb = np.maximum(lamb, 1 - lamb)
    lamb = torch.from_numpy(lamb[:, None, None, None]).float().to(x.device)
    index = torch.randperm(batch_size)
    x_mix = x * lamb + x[index] * (1 - lamb)
    y_mix = y * lamb + y[index] * (1 - lamb)
    return x_mix, y_mix


def train_tracknet(model, optimizer, data_loader, param_dict):
    model.train()
    epoch_loss = []
    for step, (_, x, y, c, _) in enumerate(tqdm(data_loader)):
        x, y = x.cuda(), y.cuda()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())
    return np.mean(epoch_loss)


def train_inpaintnet(model, optimizer, data_loader, param_dict):
    model.train()
    epoch_loss = []
    # InpaintNet 暂不记录详细 FPS，仅保持原逻辑
    for step, (_, coor_pred, coor_gt, _, vis_gt, _) in enumerate(
            tqdm(data_loader) if param_dict['verbose'] else data_loader):
        optimizer.zero_grad()
        coor_pred, coor_gt, vis_gt = coor_pred.float().cuda(), coor_gt.float().cuda(), vis_gt.float().cuda()
        # (原有的 Mask 生成逻辑...)
        # ... 省略部分未变动逻辑 ...
        return float(np.mean(epoch_loss)), 0.0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='TrackNet')
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--optim', type=str, default='Adam')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--lr_scheduler', type=str, default='')
    parser.add_argument('--bg_mode', type=str, default='')
    parser.add_argument('--alpha', type=float, default=-1)
    parser.add_argument('--frame_alpha', type=float, default=-1)
    parser.add_argument('--resume_training', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=13)
    parser.add_argument('--save_dir', type=str, default='exp')
    parser.add_argument('--exp_name', type=str, default='', help='实验名称，用于区分 logs')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=True)
    parser.add_argument('--tolerance', type=float, default=4.0, help='判定球心准确度的像素容差')
    args = parser.parse_args()
    param_dict = vars(args)

    # 路径管理
    if args.exp_name:
        args.save_dir = os.path.join(args.save_dir, f"{args.model_name}_{args.exp_name}")
    else:
        args.save_dir = os.path.join(args.save_dir, f"{args.model_name}_{time.strftime('%Y%m%d_%H%M%S')}")

    os.makedirs(args.save_dir, exist_ok=True)
    tb_writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))


    num_workers = 16
    train_dataset = Shuttlecock_Trajectory_Dataset(split='train', seq_len=args.seq_len, data_mode='heatmap',
                                                   bg_mode=args.bg_mode, debug=args.debug)
    val_dataset = Shuttlecock_Trajectory_Dataset(split='val', seq_len=args.seq_len, data_mode='heatmap',
                                                 bg_mode=args.bg_mode, debug=args.debug)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    # 模型创建与 4090 加速
    if args.model_name == 'refined':
        print(">>> 正在加载改进版 RefinedTrackNet (轻量化+运动增强)...")
        model=RefinedTrackNet(in_dim=9, out_dim=3)
    else:
        # 原有的 get_model 逻辑
        model=get_model(args.model_name, args.seq_len, args.bg_mode)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    max_val_acc = 0.
    print(f'开始训练 {args.exp_name}...')

    for epoch in range(args.epochs):
        print(f'Epoch [{epoch + 1} / {args.epochs}]')
        # 训练并获取本轮平均 FPS
        train_loss, avg_fps = train_tracknet(model, optimizer, train_loader, param_dict)
        val_loss, val_res = eval_tracknet(model, val_loader, param_dict)

        # 将 FPS 写入 TensorBoard 记录
        tb_writer.add_scalar('Performance/Average_FPS', avg_fps, epoch)
        write_to_tb(args.model_name, tb_writer, (train_loss, val_loss), val_res, epoch)

        # 保存逻辑...
        cur_val_acc = val_res['accuracy']
        if cur_val_acc >= max_val_acc:
            max_val_acc = cur_val_acc
            torch.save(model.state_dict(), os.path.join(args.save_dir, f'{args.model_name}_best.pt'))

    tb_writer.close()
    print('训练完成。')