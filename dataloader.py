import os
import copy
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.datasets as dset

from sklearn import metrics
from sklearn.metrics import accuracy_score as Acc
from sklearn.metrics import roc_auc_score as Auc
from sklearn.metrics import roc_curve as Roc
from scipy import interpolate
from scipy.special import logsumexp
import numpy as np
import pandas as pd
import shutil
import pickle
import wandb
import torchvision.transforms as transforms
import datasets as datasets


def pre_source_classes(args):
    Open = False
    if args.data == 'PACS':

        n1 = 0
        n2 = 1
        n3 = 1
        n4 = 0
        n5 = 1
        n6 = 1
        n7 = 1

        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if Open:
            print('Open Domain Split')
            # Open-Domain Split
            S123 = [i for i in range(n1)]
            S12 = [i + n1 for i in range(n2)]
            S13 = [i + n1 + n2 for i in range(n2)]
            S23 = [i + n1 + n2 * 2 for i in range(n2)]
            S1 = [i + n1 + n2 * 3 for i in range(n3)]
            S2 = [i + n1 + n2 * 3 + n3 for i in range(n3)]
            S3 = [i + n1 + n2 * 3 + n3 * 2 for i in range(n3)]

            ST1 = [S123[i] for i in range(n4)] \
                + [S12[i] for i in range(n5)] + [S13[i] for i in range(n5)] + [S23[i] for i in range(n5)] \
                + [S1[i] for i in range(n6)] + [S2[i] for i in range(n6)] + [S3[i] for i in range(n6)]
            T1 = [i + n1 + n2 * 3 + n3 * 3 for i in range(n7)]

            source_classes = [[], [], []]
            source_classes[0] = S1 + S12 + S13 + S123
            source_classes[1] = S2 + S12 + S23 + S123
            source_classes[2] = S3 + S13 + S23 + S123
            target_classes = ST1 + T1
            source_all = list(set(source_classes[0]+source_classes[1]+source_classes[2]))
            num_classes = n1 + n2 * 3 + n3 * 3
        else:
            print('Closed Domain Split')
            # Closed Domain Split
            source_classes = [[], [], [], []]
            source_classes[0] = [0,1,2,3,4,5,6]
            source_classes[1] = [0,1,2,3,4,5,6]
            source_classes[2] = [0,1,2,3,4,5,6]
            source_classes[3] = [0,1,2,3,4,5,6]
            target_classes = [0,1,2,3,4,5,6]
            ST1 = [0,1,2,3,4,5,6]
            T1 = [0,1,2,3,4,5,6]
            source_all = list(set(source_classes[0]+source_classes[1]+source_classes[2]))
            num_classes = 7

    elif args.data == 'OfficeHome':

        n1 = 3
        n2 = 6
        n3 = 11
        n4 = 1
        n5 = 2
        n6 = 3
        n7 = 11

        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_tranform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        if Open:
        # Open-Domain Split
            S123 = [i for i in range(n1)]
            S12 = [i + n1 for i in range(n2)]
            S13 = [i + n1 + n2 for i in range(n2)]
            S23 = [i + n1 + n2 * 2 for i in range(n2)]
            S1 = [i + n1 + n2 * 3 for i in range(n3)]
            S2 = [i + n1 + n2 * 3 + n3 for i in range(n3)]
            S3 = [i + n1 + n2 * 3 + n3 * 2 for i in range(n3)]

            # ST1 means the seen source classes in the target domain
            ST1 = [S123[i] for i in range(n4)] \
                + [S12[i] for i in range(n5)] + [S13[i] for i in range(n5)] + [S23[i] for i in range(n5)] \
                + [S1[i] for i in range(n6)] + [S2[i] for i in range(n6)] + [S3[i] for i in range(n6)]

            # T1 means the unseen classes in the target domain
            T1 = [i + n1 + n2 * 3 + n3 * 3 for i in range(n7)]

            source_classes = [[], [], []]
            source_classes[0] = S1 + S12 + S13 + S123
            source_classes[1] = S2 + S12 + S23 + S123
            source_classes[2] = S3 + S13 + S23 + S123
            target_classes = ST1 + T1
            source_all = list(set(source_classes[0]+source_classes[1]+source_classes[2]))
            num_classes = n1 + n2 * 3 + n3 * 3
        else:
            print('Closed Domain Split')
            # Closed Domain Split
            source_classes = [[], [], [], []]
            source_classes[0] = [i for i in range(65)]
            source_classes[1] = [i for i in range(65)]
            source_classes[2] = [i for i in range(65)]
            source_classes[3] = [i for i in range(65)]
            target_classes = [i for i in range(65)]
            ST1 = [i for i in range(65)]
            T1 = [i for i in range(65)]
            source_all = list(set(source_classes[0]+source_classes[1]+source_classes[2]))
            num_classes = 65

    elif args.data == 'OCT':

        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                                             hue=min(0.5, 0.4)),
            transforms.RandomGrayscale(0.1),
            transforms.ToTensor()
        ])
        val_tranform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        # 0: 'Normal', 
        # 1: 'AMD', 
        # 2: 'CSC', 
        # 3: 'DR', 
        # 4: 'RVO', 
        # 5: 'DME', 
        # 6: 'MEM', 
        # 7: 'MH'
        source_classes = [[], [], []]
        if args.OpenDA: 
            print('Open Domain Adapation Split')
            if args.source == 'B':
                source_classes[0] = [0, 1]
                if args.target == 'O':
                    target_classes = [0, 1, 2, 3, 4]
                elif args.target == 'S':
                    target_classes = [0, 1, 5]
                elif args.target == 'V':
                    target_classes = [0, 1, 2, 3, 4, 6, 7]
            elif args.source == 'O':
                source_classes[0] = [0, 1, 2, 3, 4]
                if args.target == 'B':
                    target_classes = [0, 1]
                elif args.target == 'S':
                    target_classes = [0, 1, 5]
                elif args.target == 'V':
                    target_classes = [0, 1, 2, 3, 4, 6, 7]
            elif args.source == 'S':
                source_classes[0] = [0, 1, 5]
                if args.target == 'B':
                    target_classes = [0, 1]
                elif args.target == 'O':
                    target_classes = [0, 1, 2, 3, 4]
                elif args.target == 'V':
                    target_classes = [0, 1, 2, 3, 4, 6, 7]
            elif args.source == 'V':
                source_classes[0] = [0, 1, 2, 3, 4, 6, 7]
                if args.target == 'B':
                    target_classes = [0, 1]
                elif args.target == 'O':
                    target_classes = [0, 1, 2, 3, 4]
                elif args.target == 'S':
                    target_classes = [0, 1, 5]

        else:
            if args.target == 'O':
                # S
                source_classes[0] = [0, 1, 5]
                # V
                source_classes[1] = [0, 1, 2, 3, 4, 6, 7]
                # B
                source_classes[2] = [0, 1]
                target_classes = [0, 1, 2, 3, 4]
            elif args.target == 'S':
                # O
                source_classes[0] = [0, 1, 2, 3, 4]
                # V
                source_classes[1] = [0, 1, 2, 3, 4, 6, 7]
                # B
                source_classes[2] = [0, 1]
                target_classes = [0, 1, 5]
            elif args.target == 'V':
                # O
                source_classes[0] = [0, 1, 2, 3, 4]
                # S
                source_classes[1] = [0, 1, 5]
                # B
                source_classes[2] = [0, 1]
                target_classes = [0, 1, 2, 3, 4, 6, 7]
            elif args.target == 'B':
                # O
                source_classes[0] = [0, 1, 2, 3, 4]
                # S
                source_classes[1] = [0, 1, 5]
                # V
                source_classes[2] = [0, 1, 2, 3, 4, 6, 7]
                target_classes = [0, 1]

        source_all = list(set(source_classes[0]+source_classes[1]+source_classes[2]))
        num_classes = len(source_all)
        # ST1 means the seen source classes in the target domain
        ST1 = [i for i in source_all if i in target_classes]
        # T1 means the unseen classes in the target domain
        T1 = [i for i in target_classes if i not in ST1]

    # They shoule be the same
    if num_classes != len(source_all):
        print('source classes error')
        exit()

    print("args.source", args.source)
    print("source_classes 0: ", source_classes[0])
    print("source_classes 1: ", source_classes[1])
    print("source_classes 2: ", source_classes[2])
    print("target_classes: ", target_classes)
    # ST1 means the seen source classes in the target domain
    print("ST1", ST1)
    # T1 means the unseen classes in the target domain
    print("T1", T1)
    print("num_classes", num_classes)
    
    return source_classes, source_all, train_transform, val_tranform, num_classes, target_classes, ST1, T1

def opendg_dataloader(source_classes, train_transform, val_tranform, ST1, target_classes, args):
    # load dataset
    dataset = datasets.__dict__[args.data]
    datasets_train, datasets_val = [], []

    if args.OpenDA:
        the_source = args.source[0]
        train_source_dataset = dataset(root=args.root, task=the_source, filter_class=source_classes[0],
                                            split='train', transform=train_transform, target_domain=args.target)
        val_source_dataset = dataset(root=args.root, task=the_source, filter_class=source_classes[0],
                                       split='val', transform=val_tranform)
    # else:
    #     for j, the_source in enumerate(args.source):
    #         train_source_dataset = dataset(root=args.root, task=the_source, filter_class=source_classes[j],
    #                                             split='train', transform=train_transform)

    #         val_source_dataset = dataset(root=args.root, task=the_source, filter_class=source_classes[j],
    #                                     split='val', transform=val_tranform)

    #         datasets_train.append(train_source_dataset)
    #         datasets_val.append(val_source_dataset)    
    #     datasets_train = ConcatDataset(datasets_train)
    #     datasets_val = ConcatDataset(datasets_val)
    
    train_loader = DataLoader(train_source_dataset, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_source_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=args.workers)

    print("Dataset size: train %d, val %d" % (len(train_loader.dataset), len(val_loader.dataset)))
    print("Dataset Weight: training weight:", train_loader.dataset.weight)

    datasets_test = []
    datasets_test_val = []
    for j, the_target in enumerate(args.target):
        print("target", the_target)
        # all the classes in the target domain
        test_dataset = dataset(root=args.root, task=the_target, filter_class=target_classes,
                                    split='all', transform=val_tranform)
        
        # only the seen source classes in the target domain
        test_val_dataset = dataset(root=args.root, task=the_target, filter_class=ST1,
                                        split='all', transform=val_tranform)

        datasets_test.append(test_dataset)
        datasets_test_val.append(test_val_dataset)
    datasets_test = ConcatDataset(datasets_test)
    datasets_test_val = ConcatDataset(datasets_test_val)

    test_loader = DataLoader(datasets_test, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers)
    test_val_loader = DataLoader(datasets_test_val, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers)

    print("Dataset size: test %d" % (len(test_loader.dataset)))
    print("Dataset size: test_val %d" % (len(test_val_loader.dataset)))
    
    return train_loader, val_loader, test_val_loader, test_loader
