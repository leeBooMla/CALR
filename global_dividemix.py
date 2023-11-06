# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
from calendar import c
from itertools import count
from math import degrees
import os.path as osp
import random
from matplotlib.font_manager import weight_dict
import numpy as np
import sys
import collections
import time
from datetime import timedelta
from collections import OrderedDict
from pandas import array
from collections import Counter

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import math

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import ClusterContrastTrainer, DivideMixTrainer
from clustercontrast.evaluators import Evaluator, extract_features, extract_features_per_cam
from clustercontrast.utils import infomap_cluster
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from clustercontrast.utils.data.sampler import GroupSampler
from clustercontrast.utils.data.preprocessor import Preprocessor
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from clustercontrast.utils.infomap_cluster import get_dist_nbr, cluster_by_infomap, get_cluster
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.visualization import evaluate_cluster
from evaluation.evaluate import evaluate_global, evaluate_local, evaluate_refine

start_epoch = best_mAP =0
CUDA_VISIBLE_DEVICES=0,1,2,3

def generate_refined_labels(fname, camid, information_node, cluster, pseudo_labels):
    #intra cam
    pseudo_labels_cam = OrderedDict()
    cluster_cam = OrderedDict()
    fname_cam = {}
    for cam in range(0,max(camid)+1):
        local_dir  = osp.join('images/market/cam'+str(cam+1), 'pseudo_label.npy')
        local_pseudo_dataset = np.load(local_dir)
        local_pseudo_dataset = local_pseudo_dataset.tolist()
        fname_cam[cam] = []
        pseudo_labels_cam[cam] = []
        for f, pid, cid in sorted(local_pseudo_dataset):
            f = f[:5]+'1'+f[5:]
            fname_cam[cam].append(f)
            pseudo_labels_cam[cam].append(pid)
        cluster_cam[cam] = get_cluster(pseudo_labels_cam[cam])
    count = 0
    for _, node in information_node.items():
        cid = camid[node]
        filename = fname[node]
        index_cam = fname_cam[cid].index(filename)
        local_cluster = cluster_cam[cid][pseudo_labels_cam[cid][index_cam]]
        global_cluster = cluster[pseudo_labels[node]]
        for nnode in global_cluster:
            if camid[nnode] != cid:
                continue
            elif camid[nnode] == cid:
                nnode_index_cam = fname_cam[cid].index(fname[nnode])
                if nnode_index_cam in local_cluster:
                    continue
                elif pseudo_labels[nnode] != -1:
                    pseudo_labels[nnode] = -1
                    count +=1
    print('remove node : {}'.format(count))
    return pseudo_labels

def generate_trainloader(args, dataset, pseudo_labels, iters):
    pseudo_labeled_dataset = []
    pseudo_unlabeled_dataset = []

    for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
        if label != -1:
            pseudo_labeled_dataset.append((fname, label.item(), cid))
        #elif label == -1:
        #    pseudo_unlabeled_dataset.append((fname, label.item(), cid))
    
    '''unlabeled_train_loader = get_train_loader(args, dataset, args.height, args.width,
                                    args.batch_size, args.workers, 0, iters,
                                    trainset=pseudo_unlabeled_dataset, no_cam=args.no_cam)

    unlabeled_train_loader.new_epoch()'''

    train_loader = get_train_loader(args, dataset, args.height, args.width,
                                    args.batch_size, args.workers, args.num_instances, iters,
                                    trainset=pseudo_labeled_dataset, no_cam=args.no_cam)

    train_loader.new_epoch()
    return train_loader#, unlabeled_train_loader

def map_pseudo_labels(pseudo_labels, num_cluster):
    if (max(pseudo_labels) + 1) == num_cluster:
        return pseudo_labels
    else:
        array = np.unique(pseudo_labels)
        for i in range(pseudo_labels.size()):
            pseudo_labels[i] = array.index(pseudo_labels[i])-1
        return pseudo_labels
    
def eval_train(args, dataset, prob, pseudo_labels, all_loss, iters):
    CE = nn.CrossEntropyLoss(reduction='none')
    pseudo_labeled_dataset = []
    pseudo_unlabeled_dataset = []
    losses = torch.zeros(20000) 
    with torch.no_grad():
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            label = torch.tensor(label)
            loss = CE(prob[i], label)
            losses[i] = loss
    losses = (losses-losses.min())/(losses.max()-losses.min())    
    all_loss.append(losses)
    input_loss = losses.reshape(-1,1)
    # fit a two-component GMM to the loss
    gmm = GaussianMixture(n_components=2,max_iter=10,tol=1e-2,reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss) 
    prob = prob[:,gmm.means_.argmin()]
    for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
        if prob[i] > 0.5:
            pseudo_labeled_dataset.append((fname, label.item(), cid))
        else:
            pseudo_unlabeled_dataset.append((fname, label.item(), cid, prob[i]))
    unlabeled_train_loader = get_train_loader(args, dataset, args.height, args.width,
                                    args.batch_size, args.workers, 0, iters,
                                    trainset=pseudo_unlabeled_dataset, no_cam=args.no_cam)

    unlabeled_train_loader.new_epoch()

    train_loader = get_train_loader(args, dataset, args.height, args.width,
                                    args.batch_size, args.workers, args.num_instances, iters,
                                    trainset=pseudo_labeled_dataset, no_cam=args.no_cam)

    train_loader.new_epoch()         
    return train_loader, unlabeled_train_loader


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_data(name, data_dir):
    root = osp.join(data_dir, name)
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(args, dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None, no_cam=False):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
    ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            #sampler = GroupSampler(train_set, num_instances, batch_size)
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
        DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                   batch_size=batch_size, num_workers=workers, sampler=sampler,
                   shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])

    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=1000, pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    warmup = 10
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load unlabeled dataset")
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = create_model(args)
    model2= create_model(args)
    # Evaluator
    evaluator = Evaluator(model)
    evaluator2 = Evaluator(model2)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    params2 = [{"params": [value]} for _, value in model2.named_parameters() if value.requires_grad]
    optimizer2 = torch.optim.Adam(params2, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    lr_scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=args.step_size, gamma=0.1)
    # Trainer
    trainer_warmup = ClusterContrastTrainer(model)
    trainer = DivideMixTrainer(model, model2)
    trainer_warmup2 = ClusterContrastTrainer(model2)
    trainer2 = DivideMixTrainer(model2, model)

    best_mAP2 = 0
    all_loss = [[],[]]
    for epoch in range(args.epochs):
        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)

            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers

        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

                        

            features, _, prob= extract_features(model, cluster_loader, print_freq=50)
            features2, _, prob2= extract_features(model2, cluster_loader, print_freq=50)
            fname = []
            camid = []
            for f, _, cid in sorted(dataset.train):
                fname.append(f)
                camid.append(cid)

            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            features2 = torch.cat([features2[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            prob = torch.cat([prob[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            prob2 = torch.cat([prob2[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

            s = time.time()
            rerank_dist = compute_jaccard_distance(features, k1=args.D_k1, k2=args.D_k2)
            rerank_dist2 = compute_jaccard_distance(features2, k1=args.D_k1, k2=args.D_k2)
            Ag = AgglomerativeClustering(n_clusters=800,
                                 affinity="precomputed",
                                 linkage='average')
            pseudo_labels = Ag.fit_predict(rerank_dist)
            pseudo_labels2 = Ag.fit_predict(rerank_dist2)
            '''features_array = F.normalize(features, dim=1).cpu().numpy()
            features_array2 = F.normalize(features2, dim=1).cpu().numpy()
            
            feat_dists, feat_nbrs = get_dist_nbr(features=features_array, k=args.k1, knn_method='faiss-gpu')
            feat_dists2, feat_nbrs2 = get_dist_nbr(features=features_array2, k=args.k1, knn_method='faiss-gpu')
            del features_array, features_array2
            _, information_node = cluster_by_infomap(feat_nbrs, feat_dists, min_sim=args.eps, cluster_num=args.k2, node_flag=1, cam_id = camid)
            _, information_node2 = cluster_by_infomap(feat_nbrs2, feat_dists2, min_sim=args.eps, cluster_num=args.k2, node_flag=1, cam_id = camid)
            cluster = get_cluster(pseudo_labels)
            cluster2 = get_cluster(pseudo_labels2)
            pseudo_labels = generate_refined_labels(fname, camid, information_node, cluster, pseudo_labels)
            pseudo_labels2 = generate_refined_labels(fname, camid, information_node2, cluster2, pseudo_labels2)
'''
            pseudo_labels = pseudo_labels.astype(np.intp)
            pseudo_labels2 = pseudo_labels2.astype(np.intp)

            print('cluster cost time: {}'.format(time.time() - s))
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
            num_cluster2 = len(set(pseudo_labels2)) - (1 if -1 in pseudo_labels2 else 0)
            #pseudo_labels = map_pseudo_labels(pseudo_labels, num_cluster)
            #pseudo_labels2 = map_pseudo_labels(pseudo_labels2, num_cluster2)
           

        cluster_features = generate_cluster_features(pseudo_labels, features)
        cluster_features2 = generate_cluster_features(pseudo_labels2, features2)

        del cluster_loader, features, features2

        # Create hybrid memory
        memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()

        memory.features = F.normalize(cluster_features, dim=1).cuda()


        memory2 = ClusterMemory(model2.module.num_features, num_cluster2, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()

        memory2.features = F.normalize(cluster_features2, dim=1).cuda()

        if epoch < warmup:
            trainer_warmup.memory = memory
            trainer_warmup2.memory = memory2
            train_loader = generate_trainloader(args, dataset, pseudo_labels, iters)
            train_loader2 = generate_trainloader(args, dataset, pseudo_labels2, iters)
            print('Warmup Net1')
            print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))
            trainer_warmup.train(epoch, train_loader, optimizer, print_freq=args.print_freq, train_iters=len(train_loader))
            print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster2))
            print('Warmup Net2')
            trainer_warmup2.train(epoch, train_loader2, optimizer2, print_freq=args.print_freq, train_iters=len(train_loader2))
        else :
            trainer.memory = memory
            trainer2.memory = memory2
            train_loader, unlabeled_train_loader = eval_train(args, dataset, prob, pseudo_labels, all_loss[0], iters)
            train_loader2, unlabeled_train_loader2 = eval_train(args, dataset, prob2, pseudo_labels2, all_loss[1], iters)
            print('Train Net1')
            print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster)) 
            trainer.train(epoch, train_loader2, unlabeled_train_loader2, optimizer, print_freq=args.print_freq, train_iters=len(train_loader))
            print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster2))
            print('Train Net2')
            trainer2.train(epoch, train_loader, unlabeled_train_loader, optimizer2, print_freq=args.print_freq, train_iters=len(train_loader2))


        if (epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)

            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))
            
            mAP2 = evaluator2.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)
            is_best2 = (mAP2 > best_mAP2)
            best_mAP2 = max(mAP2, best_mAP2)

            save_checkpoint({
                'state_dict': model2.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP2,
            }, is_best2, fpath=osp.join(args.logs_dir, 'checkpoint2.pth.tar'))

            print('\n * Finished epoch {:3d}  model2 mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP2, best_mAP2, ' *' if is_best else ''))

        lr_scheduler.step()
        lr_scheduler2.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)
    print('==> Test with the best model2:')
    checkpoint2 = load_checkpoint(osp.join(args.logs_dir, 'model_best2.pth.tar'))
    model2.load_state_dict(checkpoint2['state_dict'])
    evaluator2.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--eps', type=float, default=0.5,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--min_sim_cam', type=float, default=0.9,
                        help="min similarity for intra cam infomap")
    parser.add_argument('--eps-gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=15,
                        help="hyperparameter for KNN")
    parser.add_argument('--k2', type=int, default=4,
                        help="hyperparameter for outline")
    parser.add_argument('--D_k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--D_k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--momentum', type=float, default=0.1,
                        help="update momentum for the hybrid memory")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--step-size', type=int, default=20)

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join( '/data1/lpn/dataset'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/market/dividemix'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam', action="store_true")
    main()
