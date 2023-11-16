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
            #inputs, labels, indexes = self._parse_data(inputs)
            inputs, _, labels, cids = self._parse_data_cam(inputs)
            # forward
            '''p = float(i + epoch * train_iters) / 70 / train_iters
            alpha = 2. / (1. + np.exp(-10 * p)) - 1'''
            alpha = 1
            f_out, domain_output = self._forward(inputs, alpha)

            #prob, f_out = self._forward(inputs)
            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            loss_nce = self.memory(f_out, labels)
            domain_output = torch.softmax(domain_output, dim=1)
            err_s_domain = F.cross_entropy(domain_output, cids)


            # regularization
            '''prior = torch.ones(domain_output.size(1))/domain_output.size(1)
            prior = prior.cuda()        
            pred_mean = torch.softmax(domain_output, dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))'''

            loss = loss_nce #+ err_s_domain #+penalty
            #loss = self.memory(f_out, labels) + F.cross_entropy(prob, labels)

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
        cids = cids -1
        return imgs.cuda(), fnames, pids.cuda(), cids.cuda()

    def _forward(self, inputs, alpha):
        return self.encoder(inputs, alpha)

class DivideMixTrainer(object):
    def __init__(self, encoder, encoder2, memory=None):
        super(DivideMixTrainer, self).__init__()
        self.encoder = encoder
        self.encoder2 = encoder2
        self.memory = memory

    def train(self, epoch, labeled_trainloader,unlabeled_trainloader, optimizer, print_freq=10, train_iters=400, data_loader_cam = None):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        warm_up = 0

        end = time.time()
        num_class = 1000

        for i in range(train_iters):
            # load data
            inputs = labeled_trainloader.next()
            inputs_u = unlabeled_trainloader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)
            inputs_u,_,_,_,w_x,_ = inputs_u
            w_x = w_x.view(-1,1).type(torch.FloatTensor)

            with torch.no_grad():
                # label refinement of labeled samples
                f_out_x = self._forward(inputs)
                labels_x = labels.cpu()
                labels_x = torch.zeros(labels_x.size(0), num_class).scatter_(1, labels_x.view(-1,1), 1)
                labels_x = labels_x.cuda()
                px = torch.softmax(f_out_x, dim=1)
                px = w_x*labels_x + (1-w_x)*px
                ptx = px**(1/0.5) # temparature sharpening 
                       
                targets_x = ptx / ptx.sum(dim=1, keepdim=True) # normalize 
                targets_x = targets_x.detach()
                
                # label co-guessing of unlabeled samples
                f_out_u = self._forward(inputs_u)
                f_out_u2, _ = self.encoder2(inputs_u)

                pu = (torch.softmax(f_out_u,dim=1)+torch.softmax(f_out_u2,dim=1))/2
                ptu = pu**(1/0.5) # temparature sharpening
                targets_u = ptu / ptu.sum(dim=1, keepdim=True) # normalize
                targets_u = targets_u.detach() 

            # mixmatch
            l = np.random.beta(4, 4)        
            l = max(l, 1-l)
                    
            all_inputs = torch.cat([inputs, inputs_u], dim=0)
            all_targets = torch.cat([targets_x.cuda(), targets_u], dim=0)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]
            
            mixed_input = l * input_a + (1 - l) * input_b        
            mixed_target = l * target_a + (1 - l) * target_b
                    
            logits = self._forward(mixed_input)
            logits_x = logits[:inputs.size(0)]
            logits_u = logits[inputs.size(0):]
            logits_u = torch.softmax(logits_u,dim=1)
            mixed_target_x = mixed_target[:inputs.size(0)]
            mixed_target_u = mixed_target[inputs.size(0):]
            Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target_x, dim=1))
            Lu = torch.mean((logits_u - mixed_target_u)**2)

            # regularization
            prior = torch.ones(num_class)/num_class
            prior = prior.cuda()        
            pred_mean = torch.softmax(logits, dim=1).mean(0)
            penalty = torch.sum(prior*torch.log(prior/pred_mean))

            lamda = np.clip((epoch-warm_up)/16,0.0, 1.0)

            loss = Lx+ lamda*Lu + penalty

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
                      .format(epoch, i + 1, len(labeled_trainloader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()
    
    def _parse_data_cam(self, inputs):
        imgs, _, pids, cids, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda(), cids.cuda()

    def _forward(self, inputs):
        prob, _ = self.encoder(inputs)
        return prob

