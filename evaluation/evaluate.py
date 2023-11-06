#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import inspect

import numpy as np

from evaluation import metrics
from evaluation.utils import TextColors, Timer


def _read_meta(fn):
    labels = list()
    lb_set = set()
    with open(fn) as f:
        for lb in f.readlines():
            lb = int(lb.strip())
            labels.append(lb)
            lb_set.add(lb)
    return np.array(labels), lb_set


def evaluate(gt_labels, pred_labels, metric='pairwise'):
    if isinstance(gt_labels, str) and isinstance(pred_labels, str):
        print('[gt_labels] {}'.format(gt_labels))
        print('[pred_labels] {}'.format(pred_labels))
        gt_labels, gt_lb_set = _read_meta(gt_labels)
        pred_labels, pred_lb_set = _read_meta(pred_labels)

        print('#inst: gt({}) vs pred({})'.format(len(gt_labels),
                                                 len(pred_labels)))
        print('#cls: gt({}) vs pred({})'.format(len(gt_lb_set),
                                                len(pred_lb_set)))

    metric_func = metrics.__dict__[metric]

    result = metric_func(gt_labels, pred_labels)

    with Timer('evaluate with {}{}{}'.format(TextColors.FATAL, metric,
                                             TextColors.ENDC)):
        result = metric_func(gt_labels, pred_labels)
    if isinstance(result, np.float):
        print('{}{}: {:.4f}{}'.format(TextColors.OKGREEN, metric, result,
                                      TextColors.ENDC))
    elif metric is 'pairwise':
        ave_pre, ave_rec, fscore = result
        print('{}ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}{}'.format(
            TextColors.OKGREEN, ave_pre, ave_rec, fscore, TextColors.ENDC))
    elif metric is 'bcubed':
        ave_pre, ave_rec, fscore, expansion = result
        print('{}ave_pre: {:.4f}, ave_rec: {:.4f}, fscore: {:.4f}, expansion: {:.4f}{}'.format(
            TextColors.OKGREEN, ave_pre, ave_rec, fscore, expansion, TextColors.ENDC))
    return result

def evaluate_global(labels, pseudo_labels, precision_global, recall_global, fscore_global,precision_global_b, recall_global_b, fscore_global_b, expansion_global,nmi_global):
    pre, rec, fscore = evaluate(np.array(labels),np.array(pseudo_labels))
    precision_global.append(pre)
    recall_global.append(rec)
    fscore_global.append(fscore)
    pre, rec, fscore, expansion = evaluate(np.array(labels),np.array(pseudo_labels), metric='bcubed')
    precision_global_b.append(pre)
    recall_global_b.append(rec)
    fscore_global_b.append(fscore)
    expansion_global.append(expansion)
    nmi = evaluate(np.array(labels),np.array(pseudo_labels), metric='nmi')
    nmi_global.append(nmi)
    return precision_global, recall_global, fscore_global,precision_global_b, recall_global_b, fscore_global_b, expansion_global,nmi_global

def evaluate_local(epoch, length, labels, pseudo_labels, ave_precision_local,  ave_recall_local, ave_fscore_local, ave_precision_local_b,  ave_recall_local_b, ave_fscore_local_b, expansion_local, nmi_local):
    pre1, rec1, fscore1 = evaluate(np.array(labels), np.array(pseudo_labels))
    pre2, rec2, fscore2, expansion = evaluate(np.array(labels), np.array(pseudo_labels), metric='bcubed')
    nmi = evaluate(np.array(labels), np.array(pseudo_labels), metric='nmi')
    lamda = len(labels)/length
    if len(ave_precision_local) > epoch:
        ave_precision_local[epoch] += pre1*lamda
        ave_recall_local[epoch] += rec1*lamda
        ave_fscore_local[epoch] += fscore1*lamda
        ave_precision_local_b[epoch] += pre2*lamda
        ave_recall_local_b[epoch] += rec2*lamda
        ave_fscore_local_b[epoch] += fscore2*lamda
        expansion_local[epoch] += expansion*lamda
        nmi_local[epoch] += nmi*lamda
    else:
        ave_precision_local.append(pre1*lamda)
        ave_recall_local.append(rec1*lamda)
        ave_fscore_local.append(fscore1*lamda)
        ave_precision_local_b.append(pre2*lamda)
        ave_recall_local_b.append(rec2*lamda)
        ave_fscore_local_b.append(fscore2*lamda)
        expansion_local.append(expansion*lamda)
        nmi_local.append(nmi*lamda)
    return ave_precision_local,  ave_recall_local, ave_fscore_local, ave_precision_local_b,  ave_recall_local_b, ave_fscore_local_b, expansion_local, nmi_local

def evaluate_refine(labels, pseudo_labels,precision_global_refine, recall_global_refine, fscore_global_refine, precision_global_refine_b, recall_global_refine_b, fscore_global_refine_b, expansion_global_refine, nmi_refine):
    refine_labels, refine_pseudo_labels = [], []
    for i, label in enumerate(pseudo_labels):
        if label == -1:
            continue
        refine_labels.append(labels[i])
        refine_pseudo_labels.append(label)
    pre, rec, fscore = evaluate(np.array(refine_labels),np.array(refine_pseudo_labels))
    precision_global_refine.append(pre)
    recall_global_refine.append(rec)
    fscore_global_refine.append(fscore)
    pre, rec, fscore, expansion = evaluate(np.array(refine_labels),np.array(refine_pseudo_labels), metric='bcubed')
    precision_global_refine_b.append(pre)
    recall_global_refine_b.append(rec)
    fscore_global_refine_b.append(fscore)
    expansion_global_refine.append(expansion)
    nmi = evaluate(np.array(refine_labels),np.array(refine_pseudo_labels), metric='nmi')
    nmi_refine.append(nmi)
    return precision_global_refine, recall_global_refine, fscore_global_refine, precision_global_refine_b, recall_global_refine_b, fscore_global_refine_b, expansion_global_refine, nmi_refine

if __name__ == '__main__':
    metric_funcs = inspect.getmembers(metrics, inspect.isfunction)
    metric_names = [n for n, _ in metric_funcs]

    parser = argparse.ArgumentParser(description='Evaluate Cluster')
    parser.add_argument('--gt_labels', type=str, required=True)
    parser.add_argument('--pred_labels', type=str, required=True)
    parser.add_argument('--metric', default='pairwise', choices=metric_names)
    args = parser.parse_args()

    result = evaluate(args.gt_labels, args.pred_labels, args.metric)
