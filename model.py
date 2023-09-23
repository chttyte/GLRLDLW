import copy
import math

import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F
import os
from Utils import BulidModel
from Model import CountMeanOfFeature, CountMeanAndCovOfFeature, CountMeanOfFeatureInCluster
from  torch.nn import  BatchNorm1d

class MoPro(nn.Module):

    def __init__(self,args):
        super(MoPro, self).__init__()

        self.encoder_q = BulidModel(args)

        # model_weight_path = "../preTrainedModel/ResNet50_CropNet_trainOnSourceDomain_RAFtoFER2013.pkl"
        # model_weight_path ="/media/z/ae609a98-67c3-41ce-beed-791c5c3bf738/CD_FER_CODE/FIXMATCH_change/exp_2/CK+/all/1682884092/ResNet50_CropNet_transferToTargetDomain_RAFtoFER2013.pkl"
        # pretrained_dict = torch.load(model_weight_path)
        # model_dict = self.encoder_q.state_dict()
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # model_dict.update(pretrained_dict)
        # self.encoder_q.load_state_dict(model_dict)

        self.p_model_1 = torch.zeros((args.class_num,)).cuda()
        self.p_model_2 = torch.zeros((args.class_num,)).cuda()


        self.p_sum_1 = torch.zeros((args.class_num,)).cuda()
        self.p_sum_2 = torch.zeros((args.class_num,)).cuda()

        self.p_maxsc_1 = torch.zeros((args.class_num,)).cuda()
        self.p_maxsc_2 = torch.zeros((args.class_num,)).cuda()

        self.p_count_1 = torch.zeros((args.class_num,)).cuda()
        self.p_count_2 = torch.zeros((args.class_num,)).cuda()


        self.register_buffer("queue", torch.randn(args.low_dim, args.moco_queue))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue_full", torch.randn(384, args.moco_queue))
        self.queue_full = F.normalize(self.queue_full, dim=0)
        self.register_buffer("queue_ptr_full", torch.zeros(1, dtype=torch.long))

        self.register_buffer("prototypes", torch.zeros(args.class_num, 384))


    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    def init_threshold(self, args):

        self.p_model_1 = torch.zeros((args.class_num,)).cuda()
        self.p_model_2 = torch.zeros((args.class_num,)).cuda()

        self.p_sum_1 = torch.zeros((args.class_num,)).cuda()
        self.p_sum_2 = torch.zeros((args.class_num,)).cuda()

        self.p_maxsc_1 = torch.zeros((args.class_num,)).cuda()
        self.p_maxsc_2 = torch.zeros((args.class_num,)).cuda()


        self.p_count_1 = torch.zeros((args.class_num,)).cuda()
        self.p_count_2 = torch.zeros((args.class_num,)).cuda()


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, args):
        # gather keys before updating queue
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)

        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        # num_gpus = batch_size_all // batch_size_this
        num_gpus = 1

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        # torch.distributed.broadcast(idx_shuffle, src=0)
        #
        # # index for restoring
        # idx_unshuffle = torch.argsort(idx_shuffle)
        #
        # # shuffled index for this gpu
        # gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        # num_gpus = batch_size_all // batch_size_this
        num_gpus = 1

        # restored index for this gpu
        # gpu_idx = torch.distributed.get_rank()
        # idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x

    def print_p_model(self,):
        print("======================threshold")
        print(self.p_model_1)
        print(self.p_model_2)
        print("=======================maxscore")
        print(self.p_maxsc_1)
        print(self.p_maxsc_2)

    def show_dy_threshold(self, args, class_acc):
        p_cutoff_1 = self.p_model_1.clone().detach()
        p_cutoff_2 = self.p_model_2.clone().detach()

        p_maxsc_1 = self.p_maxsc_1.clone().detach()
        p_maxsc_2 = self.p_maxsc_2.clone().detach()
        th_1 = []
        th_2 = []
        for index in range(args.class_num):
            th_1.append((p_cutoff_1[index] + (p_maxsc_1[index] - p_cutoff_1[index]) * torch.log(class_acc[0][index] + 1.) / math.log(2) - 0.001).float())
            th_2.append((p_cutoff_2[index] + (p_maxsc_2[index] - p_cutoff_2[index]) * torch.log(class_acc[1][index] + 1.) / math.log(2) - 0.001).float())
        print('threshold is (%.3f) (%.3f) (%.3f) (%.3f) (%.3f) (%.3f) (%.3f)' % (th_1[0], th_1[1], th_1[2], th_1[3], th_1[4], th_1[5], th_1[6]))
        print('threshold is (%.3f) (%.3f) (%.3f) (%.3f) (%.3f) (%.3f) (%.3f)' % (th_2[0], th_2[1], th_2[2], th_2[3], th_2[4], th_2[5], th_2[6]))





    def forward(self, image, location, label, imgaug, args, class_acc=None, is_eval=False, is_clean=False, is_src=False, is_init_pmodel=False,p_cutoff=0.95):

        img = image.cuda(args.gpu, non_blocking=True)
        landmark = location.cuda(args.gpu, non_blocking=True)
        target = label.cuda(args.gpu, non_blocking=True)

        feat_w, logits_w  = self.encoder_q(img, landmark)

        if is_init_pmodel:

            for i in range(7):
                if i == 0 or i == 6:
                    pseudo_label = torch.softmax(logits_w[i], dim=1)
                    max_probs, max_idx = torch.max(pseudo_label, dim=-1)

                    if i == 0:
                        for max_sc, index in zip(max_probs, max_idx):
                            self.p_sum_1[index] += max_sc
                            self.p_count_1[index] += 1
                            self.p_model_1[index] = (self.p_sum_1[index] / self.p_count_1[index]).float()
                            if self.p_maxsc_1[index] < max_sc:
                                self.p_maxsc_1[index] = max_sc

                            self.p_sum_1.detach_()
                            self.p_count_1.detach_()
                            self.p_model_1.detach_()
                            self.p_maxsc_1.detach_()

                    if i == 6:
                        for max_sc, index in zip(max_probs, max_idx):
                            self.p_sum_2[index] += max_sc
                            self.p_count_2[index] += 1
                            self.p_model_2[index] = (self.p_sum_2[index] / self.p_count_2[index]).float()
                            if self.p_maxsc_2[index] < max_sc:
                                self.p_maxsc_2[index] = max_sc

                            self.p_sum_2.detach_()
                            self.p_count_2.detach_()
                            self.p_model_2.detach_()
                            self.p_maxsc_2.detach_()

            return feat_w, logits_w, target

        if is_src:
            return feat_w, logits_w

        if is_eval:
            return feat_w, logits_w, target

        if is_clean:

            img_aug = imgaug.cuda(args.gpu, non_blocking=True)
            _, logits_s = self.encoder_q(img_aug, landmark)


            unsup_loss_pos = []
            unsup_loss_neg = []
            select = []
            max_index = []

            for i in range(7):

                if i == 0 or i == 6:
                    pseudo_label = torch.softmax(logits_w[i], dim=1)
                    max_probs, max_idx = torch.max(pseudo_label, dim=-1)


                    h_label = torch.sum(-(pseudo_label * torch.log2(pseudo_label)), dim=-1)
                    w_label = 1.0 - h_label / torch.log2(torch.tensor(args.class_num).float())


                    tao = 2.0

                    w_label_pos_1 = 1 - torch.pow(class_acc[0][max_idx], 2) / tao
                    w_label_pos_2 = 1 - torch.pow(class_acc[1][max_idx], 2) / tao


                    if i == 0:
                        # mask = max_probs.ge((p_cutoff+ (1 - p_cutoff) *
                        #                     torch.log(class_acc[0][max_idx] + 1.) / math.log(2))-0.001).float()
                        mask = max_probs.ge(0.99).float()
                        select.append(max_probs.ge(0.95).long())
                        max_index.append(max_idx.long())
                        unsup_loss_pos.append((ce_loss(logits_s[i], pseudo_label) * mask * w_label_pos_1).mean())
                        # unsup_loss_pos.append((ce_loss(logits_s[i], pseudo_label) * mask).mean())

                    if i == 6:
                        # mask = max_probs.ge((p_cutoff + (1 - p_cutoff) *
                        #                     torch.log(class_acc[1][max_idx] + 1.) / math.log(2))-0.001).float()
                        mask = max_probs.ge(0.99).float()
                        select.append(max_probs.ge(0.95).long())
                        max_index.append(max_idx.long())
                        unsup_loss_pos.append((ce_loss(logits_s[i], pseudo_label) * mask * w_label_pos_2).mean())
                        # unsup_loss_pos.append((ce_loss(logits_s[i], pseudo_label) * mask).mean())


                    # select.append(max_probs.ge(p_cutoff).long())
                    # max_index.append(max_idx.long())

                    min_probs, targets_neg = torch.min(pseudo_label, dim=-1)
                    mask_neg = (pseudo_label < -0.1) * 1
                    mask_neg.scatter_(1, targets_neg.view(-1, 1), 1)
                    pred_neg = F.softmax(logits_s[i], dim=1)
                    pred_neg = 1 - pred_neg
                    pred_neg = torch.clamp(pred_neg, 1e-7, 1.0)

                    if i==0:
                        unsup_loss_neg.append(
                        ((-torch.sum(torch.log(pred_neg) * mask_neg, dim=-1)) * w_label_pos_1).mean())
                    elif i==6:
                        unsup_loss_neg.append(
                            ((-torch.sum(torch.log(pred_neg) * mask_neg, dim=-1)) * w_label_pos_2).mean())





            return unsup_loss_pos, unsup_loss_neg, select, max_index









def ce_loss(logits, targets):
    assert logits.shape == targets.shape
    log_pred = F.log_softmax(logits, dim=-1)
    nll_loss = torch.sum(-targets * log_pred, dim=1)
    return nll_loss

# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(1)]
    # torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
    # torch.gather(tensors_gather, te
    #
    #
    #
    #
    # nsor,sparse_grad=False)

    # output = torch.cat(tensor, dim=0)
    return tensor


