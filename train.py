# import common libraries
import os
import numpy as np
from os.path import join, dirname
import matplotlib.pyplot as plt
# import libraries about pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms

# logging
import wandb
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

# import from other file
from utils import fetch_scheduler, accuracy, mixup_data, mixup_criterion, js_div
from network.resnet_eis import resnet50, resnet18
from dataloader import pre_source_classes, opendg_dataloader
from OVANet import get_models
from loss import ova_loss, open_entropy, SupConLoss, RobustCrossEntropyLoss
from eval import test

# os.environ['WANDB_MODE'] = 'dryrun'
device = "cuda" if torch.cuda.is_available() else "cpu"

def ini_model(cfg, num_classes):
    # create model
    if cfg.model == "resnet18":
        model = resnet18(classes=num_classes, args=cfg)
        model = model.to(device)
        print("=> creating model '{}'".format("resnet18"))
        return model
    elif cfg.model == "resnet50":
        model = resnet50(classes=num_classes, args=cfg)
        model = model.to(device)
        print("=> creating model '{}'".format("resnet50"))
        return model
    elif cfg.model == "OVANet":
        G, C1, C2, opt_g, opt_c = get_models(cfg, num_classes)
        G = G.to(device)
        C1 = C1.to(device)
        C2 = C2.to(device)
        return G, C1, C2, opt_g, opt_c
    else:
        print("Model name is wrong")

def train_one_epoch(G, C1, C2, source_loader, target_loader, opt_g, opt_c, criterion, epoch, cfg):
    # G for the feature extractor and C1 for the closed classifier and C2 for the open classifier
    G.train()
    C1.train()
    C2.train()
    train_loss = []
    top1, n = 0., 0.

    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    max_len = max(len_train_source, len_train_target)
    con_loss = SupConLoss()
    for step in tqdm(range(max_len + 1)):
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        
        data_s = next(data_iter_s)
        data_t = next(data_iter_t)

        img_s = data_s['images']
        label = data_s['labels']
        aug_img = data_s['svdna_images']
        img_t = data_t['images']
        nearest_img_t = data_t['nearest_images']

        all_img = torch.cat([img_s, aug_img], dim=0)
        all_label = torch.cat([label, label], dim=0)

        all_img = all_img.to(device)
        all_label = all_label.to(device)
        img_t = img_t.to(device)
        nearest_img_t = nearest_img_t.to(device)
        label = label.to(device)

        opt_g.zero_grad()
        opt_c.zero_grad()
        C2.weight_norm()

        # mixed data
        mixed_x, y_a, y_b, lam = mixup_data(all_img, all_label, cfg.train.mixup_alpha)
        mixed_feat, _ = G(mixed_x)
        mixed_out_s = C1(mixed_feat)
        mixed_loss_s = mixup_criterion(criterion, mixed_out_s, y_a, y_b, lam)

        # loss for source domain
        feat, norm_feat = G(all_img)
        out_s = C1(feat)
        out_open = C2(feat)
        # source classification loss
        loss_s = criterion(out_s, all_label)
        # open set loss for source
        out_open = out_open.view(out_s.size(0), 2, -1) # [batch, 2, num_classes]
        open_loss_pos, open_loss_neg = ova_loss(out_open, all_label)
        loss_open = 0.5 * (open_loss_pos + open_loss_neg)
        
        all_loss = cfg.train.mixup_rate * mixed_loss_s + loss_s + loss_open

        # open set loss for target
        feat_t, norm_feat_t = G(img_t)
        out_open_t = C2(feat_t)
        out_open_t = out_open_t.view(img_t.size(0), 2, -1)
        ent_open = open_entropy(out_open_t)

        near_feat, _ = G(nearest_img_t)
        out_open_near = C2(near_feat)
        out_open_near = out_open_near.view(nearest_img_t.size(0), 2, -1)
        ent_open_near = open_entropy(out_open_near)

        ent_open = 0.5 * (ent_open + ent_open_near)

        # consistenty regularization
        out_open_t = F.softmax(out_open_t, 1)
        out_open_near = F.softmax(out_open_near, 1)
        L_nscr = torch.mean(torch.sum(torch.sum(torch.abs(
            out_open_t - out_open_near)**2, 1), 1))
        
        all_loss = all_loss + cfg.train.ent_ratio * ent_open + cfg.train.nscr_ratio * L_nscr
        
        all_loss.backward()
        opt_g.step()
        opt_c.step()

        # measure accuracy
        acc1, acc2 = accuracy(out_s, all_label, topk=(1, 2))
        top1 += acc1
        n += all_img.size(0)

        # Record the loss and accuracy.
        train_loss.append(all_loss.item())

    top1 = (top1 / n) * 100
    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = top1

    return G, C1, C2, train_loss, train_acc

def valid(G, C1, C2, loader, criterion, cfg):

    G.eval()
    C1.eval()
    C2.eval()
    valid_loss = []
    top1, n = 0., 0.
    for output in tqdm(loader):
        img = output['images']
        labels = output['labels']

        img = img.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            feat, norm_feat = G(img)
            pred = C1(feat)
            loss = criterion(pred, labels)

            # measure accuracy
            acc1, acc2 = accuracy(pred, labels, topk=(1, 2))
            top1 += acc1
            n += img.size(0)

        valid_loss.append(loss.item())

    top1 = (top1 / n) * 100
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = top1

    return valid_loss, valid_acc

def main_worker(train_loader, val_loader, test_val_loader, test_loader, val_tranform, num_classes, target_classes, ST1, T1, cfg):

    # Initialize model
    G, C1, C2, opt_g, opt_c = ini_model(cfg, num_classes)
    
    # scheduler
    scheduler_g = fetch_scheduler(cfg, opt_g)
    scheduler_c = fetch_scheduler(cfg, opt_c)
    
    # criterion
    criterion = RobustCrossEntropyLoss(weight=train_loader.dataset.weight, label_smoothing=0.1)
    val_criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_test_acc = 0.0
    num_epoch = cfg.num_epoch

    for epoch in range(num_epoch):
        # train
        G, C1, C2, train_loss, train_acc = train_one_epoch(G, C1, C2, train_loader, test_loader, opt_g, opt_c, criterion, epoch, cfg)
        print(f"[ Train | {epoch + 1:03d}/{num_epoch:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        # learning rate decay
        scheduler_g.step()
        scheduler_c.step()
        realLR_g = scheduler_g.get_last_lr()[0]
        realLR_c = scheduler_c.get_last_lr()[0]

        # valid
        test_val_loss, test_val_acc = valid(G, C1, C2, test_val_loader, val_criterion, cfg)
        print(f"[ Valid | {epoch + 1:03d}/{num_epoch:03d} ] loss = {test_val_loss:.5f}, acc = {test_val_acc:.5f}")

        # test
        acc_all, h_score, known_acc, unknown = test(epoch+1, test_loader, num_classes, G, [C1, C2], open=True)
        print("acc all %s h_score %s " % (acc_all, h_score))
        # wandb
        wandb.log(step=epoch+1, data={'epoch': epoch + 1,
            'train/train_loss': train_loss, 'train/train_acc': train_acc, 'train/lr_g': realLR_g, 'train/lr_c': realLR_c,
            'test/test_acc': acc_all, 'test/test_h_score': h_score,  
            'test/test_known_acc': known_acc, 'test/test_unknown_acc': unknown,
            'val/test_val_acc': test_val_acc, 'val/test_val_loss': test_val_loss,}) 

        if epoch > 0 and epoch % 10 == 0:
            model_name = './checkpoints/' + cfg.model_name + '_epoch_' + str(epoch) + '.pt'
            model_path = os.path.join(model_name)

            save_dic = {
                'g_state_dict': G.state_dict(),
                'c1_state_dict': C1.state_dict(),
                'c2_state_dict': C2.state_dict(),
            }
            torch.save(save_dic, model_path)
            print('saving model at epoch {}'.format(epoch))
            print('The test acc is {:.3f}'.format(acc_all))

def wandb_init(cfg: DictConfig):
    wandb.init(
        project='Open_OCT',
        entity='nekokiku',
        name=cfg.exp_name,
        notes=cfg.exp_desc,
        save_code=True,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    OmegaConf.save(config=cfg, f=os.path.join(wandb.run.dir, 'conf.yaml'))

@hydra.main(version_base='1.2', config_path='configs', config_name='opendg')
def main(cfg):
    # print config
    print(OmegaConf.to_yaml(cfg))
    wandb_init(cfg)

    source_classes, source_all, train_transform, val_tranform, num_classes, target_classes, ST1, T1 = pre_source_classes(cfg)
    train_loader, val_loader, test_val_loader, test_loader = opendg_dataloader(source_classes, train_transform, val_tranform, ST1, target_classes, cfg)
    main_worker(train_loader, val_loader, test_val_loader, test_loader, val_tranform,num_classes, target_classes, ST1, T1, cfg)

    # do this after training
    wandb.finish()

if __name__ == '__main__':
    main()

