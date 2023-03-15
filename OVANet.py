import torch
from torch import nn
import torchvision.transforms as transforms
import torch.optim as optim
from network.basenet import ResClassifier_MME, ResBase, VGGBase

def get_model_mme(net, top=False):
    dim = 2048
    if "resnet" in net:
        model_g = ResBase(net, top=top)
        if "resnet18" in net:
            dim = 512
        if net == "resnet34":
            dim = 512
    elif "vgg" in net:
        model_g = VGGBase(option=net, pret=True, top=top)
        dim = 4096
    if top:
        dim = 1000
    print("selected network %s"%net)
    return model_g, dim

def get_models(cfg, num_class):
    net = 'resnet50'
    G, dim = get_model_mme(net)

    C1 = ResClassifier_MME(num_classes=num_class,
                           norm=False, input_size=dim)
    C2 = ResClassifier_MME(num_classes=2 * num_class,
                           norm=False, input_size=dim)

    params = []
    if net == "vgg16":
        for key, value in dict(G.named_parameters()).items():
            if 'classifier' in key:
                params += [{'params': [value], 'lr': cfg.train.multi,
                            'weight_decay': cfg.train.weight_decay}]

    else:
        for key, value in dict(G.named_parameters()).items():

            if 'bias' in key:
                params += [{'params': [value], 'lr': cfg.train.multi,
                            'weight_decay': cfg.train.weight_decay}]
            else:
                params += [{'params': [value], 'lr': cfg.train.multi,
                            'weight_decay': cfg.train.weight_decay}]

    opt_g = optim.SGD(G.parameters(), momentum=cfg.train.sgd_momentum, lr=cfg.train.lr,
                      weight_decay=0.0005, nesterov=True)
    opt_c = optim.SGD(list(C1.parameters()) + list(C2.parameters()), lr=cfg.train.lr,
                       momentum=cfg.train.sgd_momentum, weight_decay=0.0005,
                       nesterov=True)

    # opt_g = optim.AdamW(G.parameters(), lr=cfg.learning_rate)
    # opt_c = optim.AdamW(list(C1.parameters()) + list(C2.parameters()), lr=cfg.learning_rate)

    return G, C1, C2, opt_g, opt_c

