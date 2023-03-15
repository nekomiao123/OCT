import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import cv2
from PIL import Image
from torch.optim import lr_scheduler
import torchvision.transforms as transforms

def fetch_scheduler(cfg, optimizer):
    if cfg.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=cfg.T_max, 
                                                   eta_min=cfg.min_lr)
    elif cfg.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=cfg.T_0, 
                                                             eta_min=cfg.min_lr)
    elif cfg.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=cfg.min_lr,)
    elif cfg.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif cfg.scheduler == 'WarmupCosineLR':
        warm_up_epochs = cfg.warm_up
        num_epoch = cfg.num_epoch
        warm_up_with_cosine_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else 0.5 * ( math.cos((epoch - warm_up_epochs) /(num_epoch - warm_up_epochs) * math.pi) + 1)
        scheduler = lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_cosine_lr)
    elif cfg.scheduler == 'WarmupInvLR':
        warm_up_epochs = cfg.warm_up
        num_epoch = cfg.num_epoch
        power=0.75
        gamma=10
        warm_up_with_inv_lr = lambda epoch: epoch / warm_up_epochs if epoch <= warm_up_epochs else (1 + gamma * min(1.0, (epoch - warm_up_epochs) /(num_epoch - warm_up_epochs))) ** (-power)
        scheduler = lr_scheduler.LambdaLR( optimizer, lr_lambda=warm_up_with_inv_lr)

    return scheduler

def mixup_data(x, y, alpha=1.0):
    # 对数据的mixup 操作 x = lambda*x_i+(1-lamdda)*x_j
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
 
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
 
    mixed_x = lam * x + (1 - lam) * x[index, :]    # 此处是对数据x_i 进行操作
    y_a, y_b = y, y[index]    # 记录下y_i 和y_j
    return mixed_x, y_a, y_b, lam    # 返回y_i 和y_j 以及lambda

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    # 对loss函数进行混合，criterion是crossEntropy函数
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def js_div(p_output, q_output, get_softmax=False):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output))/2


def maybe_dictionarize(batch):
    if isinstance(batch, dict):
        return batch

    if len(batch) == 2:
        batch = {'images': batch[0], 'labels': batch[1]}
    elif len(batch) == 3:
        batch = {'images': batch[0], 'labels': batch[1], 'metadata': batch[2]}
    else:
        raise ValueError(f'Unexpected number of elements: {len(batch)}')

    return batch



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def get_train_standard_transformers():
    img_tr = [transforms.RandomResizedCrop((224,224), (0.8, 1.0))]
    img_tr.append(transforms.RandomHorizontalFlip(0.5))
    img_tr.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                                             hue=min(0.5, 0.4)))

    tile_tr = []
    tile_tr.append(transforms.RandomGrayscale(0.1))
    tile_tr = tile_tr + [transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)

def get_val_transformer():
    img_tr = [transforms.Resize((224,224)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)