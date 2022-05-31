import numpy as np
import os


CAT_LIST = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train',
        'tvmonitor', 'meanIOU']

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    ##  The 5-th cls is exclude.
    n_class = n_class+1
    hist = np.zeros((n_class, n_class))

    for lt, lp in zip(label_trues, label_preds):
        lp[lt==4]=4
        tmp = _fast_hist(lt.flatten(), lp.flatten(), n_class)
        hist += tmp
    hist[4,4]=0
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist)[0:4] / hist.sum(axis=1)[0:4]
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist)[0:4] / ((hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))[0:4]) #true和pred都没有，则nan；true有pred没有或者true没有pred有，则zero
    mean_iu = np.nanmean(iu) #gt中有的其实就不会存在nan了
    freq = hist.sum(axis=1)[0:4] / hist.sum() #groundtrue中每一个类的占比
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum() #groundtrue中不存在的不考虑
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc, #不存在的像素类别也考虑进去了，算进分母；直接求所有像素的分类均值
        "Mean Accuracy": acc_cls, #不存在的像素类别不考虑，先算每一类的准确率，再求所有类的平均值；结果受每个类别的像素点占比影响较大；groundtrue中不存在的类不考虑
        "Frequency Weighted IoU": fwavacc, #只考虑了gt中存在的类
        "Mean IoU": mean_iu, #只考虑了gt中存在的类；算均值结果受每个类别的像素点占比影响较大
        "Class IoU": cls_iu,
    }

