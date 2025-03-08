from collections import namedtuple
cfg = {
    'train_dir':'~/datasets/openimages/train',
    'save_dir' :'../test/train',
    'resume': '',
    'lr': 1e-4,
    'batch_size': 12,
    'crop_size': 128,
    'epoch': 10,
    'log_interval': 100,
    'decay_rate': 0.75,
    'decay_interval': 5,
    'seed': 22,
    'prior_ar_width': 32,
    'prior_ar_depth': 3,
    'master_port': '12342',
    'num_gpus': 2,
}
args = namedtuple('args', cfg.keys())(*cfg.values())

import os
import logging
import datetime

import torch as th
import numpy as np
import random

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
import torch.multiprocessing as mp
from torchvision.transforms import Compose, RandomCrop, ToTensor
from models.prefitter import Prefitter, PrefitterParameter

def get_time():
    return str(datetime.datetime.now())[0:19]

def reduce_mean(tensor, world_size):
    if isinstance(tensor, th.Tensor):
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= world_size
    else:
        rt = tensor
    return rt

def set_logger(log):
    level = getattr(logging, 'INFO', None)
    handler = logging.FileHandler(log)
    formatter = logging.Formatter('')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(level)

def seed_all(seed):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main():
    prefix = f'{get_time()[:10]}'
    if not os.path.exists(os.path.join(args.save_dir, prefix)):
         os.mkdir(os.path.join(args.save_dir, prefix))
    log_dir = os.path.join(args.save_dir, prefix, 'log')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    ckpt_dir = os.path.join(args.save_dir, prefix, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)
    
    world_size = args.num_gpus
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(_single_task, args=(world_size, log_dir, ckpt_dir), nprocs=world_size)

def _single_task(rank, world_size, log_dir, ckpt_dir):
    os.environ['RANK'] = str(rank)
    th.cuda.set_device(rank)
    device = th.device('cuda', rank)
    is0 = rank == 0
    dist.init_process_group(backend='nccl', world_size=world_size, init_method='env://')

    train_set = ImageFolder(args.train_dir, transform=Compose([RandomCrop(args.crop_size),ToTensor()]))
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler, pin_memory=True, shuffle=False)

    net_parameters = PrefitterParameter(
        prior_arm_depth=args.prior_ar_depth,
        prior_arm_width=args.prior_ar_width
    )
    net = Prefitter(net_parameters)

    ckpt_path = os.path.join(args.resume, 'ckpt.pth')
    if os.path.exists(ckpt_path):
        net.load_state_dict(th.load(ckpt_path,map_location='cpu'))
    net.to_device(device)
    net = DDP(net, device_ids=[rank])
    
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    net = net.to(device)
    # load optimizer
    optimizer = th.optim.Adam(net.parameters(), lr=args.lr)
    optm_path = os.path.join(args.resume, 'optm.pth')
    if os.path.exists(optm_path):
        optimizer.load_state_dict(th.load(optm_path, map_location='cpu'))
    
    # load scheduler
    lr_scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_rate)
    lr_path = os.path.join(args.resume, 'lr.pth')
    if os.path.exists(lr_path):
        lr_scheduler.load_state_dict(th.load(lr_path, map_location='cpu'))
    if is0:
        set_logger(os.path.join(log_dir, 'train.log'))
        logging.info(args)
        logging.info(f'use {world_size} gpus')
    net.train()
    # train
    losses = []
    for epoch_id in range(args.epoch):
        train_sampler.set_epoch(epoch_id)
        for idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            loss = net(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if idx % args.log_interval == 0 and is0:
                logging.info(f'{get_time()} |epoch {epoch_id} | train | {idx} | {np.mean(losses):.4f}')
                losses = []
        if is0:
            th.save(net.module.state_dict(), os.path.join(ckpt_dir, f'ckpt.pth'))
            th.save(optimizer.state_dict(), os.path.join(ckpt_dir, f'optm.pth'))
            th.save(lr_scheduler.state_dict(), os.path.join(ckpt_dir, f'lr.pth'))
            logging.info(f'ckpt saved to {ckpt_dir}')
        lr_scheduler.step()

if __name__ == '__main__':
    main()
