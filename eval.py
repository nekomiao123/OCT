import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import os
import logging
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_auc_score,  accuracy_score

def test(step, dataset_test, n_share, G, Cs,
         open_class = None, open=False, entropy=False, thr=None):
    G.eval()
    for c in Cs:
        c.eval()
    ## Known Score Calculation.
    correct = 0
    correct_close = 0
    size = 0
    per_class_num = np.zeros((n_share + 1))
    per_class_correct = np.zeros((n_share + 1)).astype(np.float32)
    class_list = [i for i in range(n_share)]
    for batch_idx, data in enumerate(dataset_test):
        with torch.no_grad():
            img_t = data['images']
            label_t = data['labels']
            img_t = img_t.cuda()
            label_t = label_t.cuda()
            # make open class == label n
            outlier_flag = (label_t > (n_share - 1)).float()
            label_t = label_t * (1 - outlier_flag) + n_share * outlier_flag

            feat, norm_feat = G(img_t)
            out_t = Cs[0](feat)
            if batch_idx == 0:
                open_class = int(out_t.size(1)) # open_class = n 
                class_list.append(open_class) 
            pred = out_t.data.max(1)[1]
            correct_close += pred.eq(label_t.data).cpu().sum()
            # print("pred", pred)
            # print("label_t", label_t)
            # print("correct_close", correct_close)
            out_t = F.softmax(out_t, dim=1)
            entr = -torch.sum(out_t * torch.log(out_t), 1).data.cpu().numpy()
            if entropy:
                pred_unk = -torch.sum(out_t * torch.log(out_t), 1)
                ind_unk = np.where(entr > thr)[0]
            else:
                out_open = Cs[1](feat)
                out_open = F.softmax(out_open.view(out_t.size(0), 2, -1),1)
                tmp_range = torch.arange(0, out_t.size(0)).long().cuda()
                pred_unk = out_open[tmp_range, 0, pred] # [batch_size,]
                
                # direct method
                # ind_unk = np.where(pred_unk.data.cpu().numpy() > 0.5)[0] # id of unknown > 0.5
                # pred[ind_unk] = open_class     # if unknown > 0.5, pred = open_class    
                
                # C + 1 method
                pred_known = out_open[:,1,:].squeeze() # B, C
                score = pred_known * out_t # B, C
                unk_score = (1 - score.sum(1)).unsqueeze(1) # B, 1
                scores = torch.cat([score, unk_score], dim=-1) # B, C + 1
                label = torch.argmax(scores, dim=-1) # B,
                pred[label == open_class] = open_class

            correct += pred.eq(label_t.data).cpu().sum()
            # print("new pred", pred)
            # print("correct", correct)
            # print("------------------")
            pred = pred.cpu().numpy()
            k = label_t.data.size()[0]
            for i, t in enumerate(class_list):
                t_ind = np.where(label_t.data.cpu().numpy() == t)
                correct_ind = np.where(pred[t_ind[0]] == t)
                per_class_correct[i] += float(len(correct_ind[0]))
                per_class_num[i] += float(len(t_ind[0]))
            size += k
            if open:
                label_t = label_t.data.cpu().numpy()
                if batch_idx == 0:
                    label_all = label_t
                    pred_open = pred_unk.data.cpu().numpy()
                    pred_all = out_t.data.cpu().numpy()
                    pred_ent = entr
                else:
                    pred_open = np.r_[pred_open, pred_unk.data.cpu().numpy()]
                    pred_ent = np.r_[pred_ent, entr]
                    pred_all = np.r_[pred_all, out_t.data.cpu().numpy()]
                    label_all = np.r_[label_all, label_t]
    if open:
        Y_test = label_binarize(label_all, classes=[i for i in class_list])
        # roc = roc_auc_score(Y_test[:, -1], pred_open)
        # roc_ent = roc_auc_score(Y_test[:, -1], pred_ent)
        # roc_softmax = roc_auc_score(Y_test[:, -1], -np.max(pred_all, axis=1))
        roc = 0.0
        roc_ent = 0.0
        roc_softmax = 0.0
        ## compute best h-score by grid search. Note that we compupte
        ## this score just to see the difference between learned threshold
        ## and best one.
        best_th, best_h, mean_score = select_threshold(pred_all, pred_open,
                                                         label_all, class_list)
    else:
        roc = 0.0
        roc_ent = 0.0
        best_th = 0.
        best_h = 0.
        roc_softmax = 0.0
    per_class_acc = per_class_correct / per_class_num
    acc_all = 100. * float(correct) / float(size)
    close_count = float(per_class_num[:len(class_list) - 1].sum())
    acc_close_all = 100. * float(correct_close) / close_count
    known_acc = per_class_acc[:len(class_list)-1].mean()
    unknown = per_class_acc[-1]
    h_score = 200 * known_acc * unknown / (known_acc + unknown)
    output = ["step %s"%step,
              "closed perclass", list(per_class_acc),
              "acc per class %s"%(float(per_class_acc.mean())),
              "acc all%s" % float(acc_all),
              "acc close all %s" % float(acc_close_all),
              "known acc %s" % float(known_acc),
              "unknown acc %s" % float(unknown),
              "h score %s" % float(h_score),
              "roc %s"% float(roc),
              "roc ent %s"% float(roc_ent),
              "roc softmax %s"% float(roc_softmax),
              "best hscore %s"%float(best_h),
              "best thr %s"%float(best_th)]
    print(output)
    return acc_all, h_score, known_acc, unknown


def select_threshold(pred_all, conf_thr, label_all,
                     class_list, thr=None):
    num_class  = class_list[-1]
    best_th = 0.0
    best_f = 0
    best_known = 0
    best_unknown = 0
    if thr is not None:
        pred_class = pred_all.argmax(axis=1)
        ind_unk = np.where(conf_thr > thr)[0]
        pred_class[ind_unk] = num_class
        return accuracy_score(label_all, pred_class), \
               accuracy_score(label_all, pred_class), \
               accuracy_score(label_all, pred_class)
    ran = np.linspace(0.0, 1.0, num=20)
    conf_thr = conf_thr / conf_thr.max()
    scores = []
    for th in ran:
        pred_class = pred_all.argmax(axis=1)
        ind_unk = np.where(conf_thr > th)[0]
        pred_class[ind_unk] = num_class
        score, known, unknown = h_score_compute(label_all, pred_class,
                                                class_list)
        scores.append(score)
        if score > best_f:
            best_th = th
            best_f = score
            best_known = known
            best_unknown = unknown
    mean_score = np.array(scores).mean()
    print("best known %s best unknown %s "
          "best h-score %s"%(best_known, best_unknown, best_f))
    return best_th, best_f, mean_score


def h_score_compute(label_all, pred_class, class_list):
    per_class_num = np.zeros((len(class_list)))
    per_class_correct = np.zeros((len(class_list))).astype(np.float32)
    for i, t in enumerate(class_list):
        t_ind = np.where(label_all == t)
        correct_ind = np.where(pred_class[t_ind[0]] == t)
        per_class_correct[i] += float(len(correct_ind[0]))
        per_class_num[i] += float(len(t_ind[0]))
    open_class = len(class_list)
    per_class_acc = per_class_correct / per_class_num
    known_acc = per_class_acc[:open_class - 1].mean()
    unknown = per_class_acc[-1]
    h_score = 2 * known_acc * unknown / (known_acc + unknown)
    return h_score, known_acc, unknown
