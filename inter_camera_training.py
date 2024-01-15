# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import
import argparse
from calendar import c
from itertools import count
from math import degrees
import os.path as osp
import os
import random
from matplotlib.font_manager import weight_dict
import numpy as np
import sys
import collections
import time
from datetime import timedelta
from collections import OrderedDict, Counter
from pandas import array
from collections import Counter

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import math

from calr import datasets
from calr import models
from calr.models.cm import ClusterMemory
from calr.trainers import ClusterContrastTrainer
from calr.evaluators import Evaluator, extract_features, extract_features_per_cam
from calr.utils import infomap_cluster
from calr.utils.data import IterLoader
from calr.utils.data import transforms as T
from calr.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from calr.utils.data.sampler import GroupSampler
from calr.utils.data.preprocessor import Preprocessor
from calr.utils.logging import Logger
from calr.utils.serialization import load_checkpoint, save_checkpoint
from calr.utils.infomap_cluster import get_dist_nbr, cluster_by_infomap, get_cluster
from calr.utils.faiss_rerank import compute_jaccard_distance
from evaluation.evaluate import evaluate_global, evaluate_local, evaluate_refine

start_epoch = best_mAP = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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


def create_model(args, num_cams):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout,
                          num_classes=0, pooling_type=args.pooling_type, num_cams=num_cams)
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
    model = create_model(args, dataset.num_train_cams)
    # Evaluator
    evaluator = Evaluator(model)

    # Optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    # Trainer
    trainer = ClusterContrastTrainer(model)

    for epoch in range(args.epochs):
        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(pseudo_labels, features):
            centers = collections.defaultdict(list)

            for i, label in enumerate(pseudo_labels):
                if label == -1:
                    continue
                centers[pseudo_labels[i]].append(features[i])

            centers = [
                torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
            ]

            centers = torch.stack(centers, dim=0)
            return centers
        
        def generate_cluster_weights(pseudo_labels, dataset):
            cluster_cid = collections.defaultdict(list)
            weights = torch.ones(max(pseudo_labels)+1)

            for i, ((_, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
                if label != -1:
                    cluster_cid[label].append(cid)

            for idx in sorted(cluster_cid.keys()):
                camera_num = torch.tensor(list(Counter(cluster_cid[idx]).values()), dtype=torch.float64)
                weight = camera_num/torch.sum(camera_num)
                weight = -torch.sum(weight*torch.log(weight))
                weight = torch.log(weight+1)
                weights[idx] = weight

            return weights

        with torch.no_grad():
            print('==> Create pseudo labels for unlabeled data')
            cluster_loader = get_test_loader(dataset, args.height, args.width,
                                             args.batch_size, args.workers, testset=sorted(dataset.train))

                        

            features, _= extract_features(model, cluster_loader, print_freq=50)
            fname = []
            camid = []
            labels = []
            for f, pid, cid in sorted(dataset.train):
                fname.append(f)
                labels.append(pid)
                camid.append(cid)

            features = torch.cat([features[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
            features_array = F.normalize(features, dim=1).cpu().numpy()
            s = time.time()
            feat_dists, feat_nbrs = get_dist_nbr(features=features_array, k=args.k1, knn_method='faiss-gpu')
            pseudo_labels, infomation_node = cluster_by_infomap(feat_nbrs, feat_dists, min_sim=args.eps, cluster_num=args.k2, node_flag=1, cam_id = camid)
            cluster = get_cluster(pseudo_labels)
            if 1:
                #intra cam
                pseudo_labels_cam = OrderedDict()
                cluster_cam = OrderedDict()
                fname_cam = {}
                per_cam_labels = {}
                for cam in range(0,max(camid)+1):
                    local_dir  = osp.join(args.local_dir+str(cam+1), 'pseudo_label.npy')
                    local_pseudo_dataset = np.load(local_dir)
                    local_pseudo_dataset = local_pseudo_dataset.tolist()
                    fname_cam[cam] = []
                    pseudo_labels_cam[cam] = []
                    for f, pid, cid in sorted(local_pseudo_dataset):
                        f = f[:5]+'1'+f[6:]
                        fname_cam[cam].append(f)
                        pseudo_labels_cam[cam].append(pid)
                    cluster_cam[cam] = get_cluster(pseudo_labels_cam[cam])
                    per_cam_labels[cam] = []
                    for fname_l in fname_cam[cam]:
                        per_cam_labels[cam].append(labels[fname.index(fname_l)])
                count = 0
                #decay = 0.5*(1+math.cos(math.pi*epoch/(3*args.epochs)))
                decay = 1
                for label, node in infomation_node.items():
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
                                r = random.random()
                                if r <= decay:
                                    pseudo_labels[nnode] = -1
                                    count +=1
                print('remove node : {}'.format(count))
            pseudo_labels = pseudo_labels.astype(np.intp)

            print('cluster cost time: {}'.format(time.time() - s))
            num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)


        cluster_features = generate_cluster_features(pseudo_labels, features)
        cluster_weights = generate_cluster_weights(pseudo_labels, dataset)

        del cluster_loader, features

        # Create hybrid memory
        memory = ClusterMemory(model.module.num_features, num_cluster, temp=args.temp,
                               momentum=args.momentum, use_hard=args.use_hard).cuda()

        memory.features = F.normalize(cluster_features, dim=1).cuda()
        #memory.weights = cluster_weights.cuda()
        #warm up weights
        if epoch < 5:
            memory.weights = None
        else :
            memory.weights = cluster_weights.cuda()
        trainer.memory = memory
        pseudo_labeled_dataset = []

        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            if label != -1:
                pseudo_labeled_dataset.append((fname, label.item(), cid))

        print('==> Statistics for epoch {}: {} clusters'.format(epoch, num_cluster))

        train_loader = get_train_loader(args, dataset, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters,
                                        trainset=pseudo_labeled_dataset, no_cam=args.no_cam)

        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, train_iters=len(train_loader))

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

        lr_scheduler.step()

    print('==> Test with the best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-paced contrastive learning on unsupervised re-ID")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='veri',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=224, help="input height")
    parser.add_argument('--width', type=int, default=224, help="input width")
    parser.add_argument('--num-instances', type=int, default=16,
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
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--temp', type=float, default=0.05,
                        help="temperature for scaling contrastive loss")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join( '/data1/lpn/dataset'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/veri'))
    parser.add_argument('--local_dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'local results/veri/cam'))
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam', action="store_true")
    main()
