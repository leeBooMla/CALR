from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
import torch
import torch.nn.functional as F
import numpy as np


class ClusterContrastTrainer(object):
    def __init__(self, encoder, memory=None):
        super(ClusterContrastTrainer, self).__init__()
        self.encoder = encoder
        self.memory = memory

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400, data_loader_cam = None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, _, labels, cids = self._parse_data_cam(inputs)
            # forward
            CA_flag=1
            f_out, domain_output = self._forward(inputs, CA_flag)

            loss_nce = self.memory(f_out, labels)
            domain_output = torch.softmax(domain_output, dim=1)
            err_s_domain = F.cross_entropy(domain_output, cids)

            if epoch < 3:
                loss = loss_nce + err_s_domain
            else :
                loss = loss_nce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()
    
    def _parse_data_cam(self, inputs):
        imgs, fnames, pids, cids, _ = inputs
        return imgs.cuda(), fnames, pids.cuda(), cids.cuda()

    def _forward(self, inputs, CA_flag):
        return self.encoder(inputs, CA_flag)