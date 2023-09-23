import os
import sys
import time
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter

from Loss import HAFN, SAFN
from Utils import *
from MoPro import *

def construct_args():
    parser = argparse.ArgumentParser(description='Expression Classification Training')

    parser.add_argument('--Log_Name', type=str,default='ResNet50_CropNet_GCNwithIntraMatrixAndInterMatrix_useCluster_withoutAFN_trainOnSourceDomain_RAFtoAFED', help='Log Name')
    parser.add_argument('--OutputPath', type=str,default='.', help='Output Path')
    parser.add_argument('--Backbone', type=str, default='ResNet50', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet']) # 挑选backbone
    parser.add_argument('--Resume_Model', type=str, help='Resume_Model', default='None') # 导入pretrained模型用的
    parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')  # 选择的gpu型号

    parser.add_argument('--useAFN', type=str2bool, default=False, help='whether to use AFN Loss')
    parser.add_argument('--methodOfAFN', type=str, default='SAFN', choices=['HAFN', 'SAFN']) #  AFN --  Adaptive Feature Norm 一种特征的自适应方法
    parser.add_argument('--radius', type=float, default=40, help='radius of HAFN (default: 25.0)') # k-means计算的半径
    parser.add_argument('--deltaRadius', type=float, default=0.001, help='radius of SAFN (default: 1.0)') # ! 这个跟上面那个半径有什么区别
    parser.add_argument('--weight_L2norm', type=float, default=0.05, help='weight L2 norm of AFN (default: 0.05)') # AFN是一种求损失的方法

    # ! dataset
    parser.add_argument('--faceScale', type=int, default=112, help='Scale of face (default: 112)') # 人脸图片的尺寸
    parser.add_argument('--sourceDataset', type=str, default='AFED', choices=['RAF', 'AFED', 'MMI', 'FER2013']) # source dataset的名字
    parser.add_argument('--targetDataset', type=str, default='JAFFE', choices=['RAF', 'CK+', 'JAFFE', 'MMI', 'Oulu-CASIA', 'SFEW', 'FER2013', 'ExpW', 'AFED', 'WFED']) # 目标域
    parser.add_argument('--train_batch_size', type=int, default=32, help='input batch size for training (default: 64)') # 训练的batch size
    parser.add_argument('--test_batch_size', type=int, default=32, help='input batch size for testing (default: 64)')   # 测试集的batch size
    parser.add_argument('--useMultiDatasets', type=str2bool, default=False, help='whether to use MultiDataset')         #  是否使用多个数据集（使用多个数据集的什么意思呢）

    parser.add_argument('--lr', type=float, default=0.0001) # 学习率
    parser.add_argument('--epochs', type=int, default=60,help='number of epochs to train (default: 10)') # 训练代数
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)')       # 动量 
    parser.add_argument('--weight_decay', type=float, default=0.0001,help='SGD weight decay (default: 0.0005)') # 正则项系数

    parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model') # 是否只是进行测试，意思就是不用训练
    parser.add_argument('--showFeature', type=str2bool, default=True, help='whether to show feature') # 展示特征（咩啊，用来干嘛的，展示特征图把）

    parser.add_argument('--useIntraGCN', type=str2bool, default=True, help='whether to use Intra-GCN') #  在域内是否使用GCN传播
    parser.add_argument('--useInterGCN', type=str2bool, default=True, help='whether to use Inter-GCN') #  在域间是否使用GCN传播
    parser.add_argument('--useLocalFeature', type=str2bool, default=True, help='whether to use Local Feature') # 是否使用局部特征

    parser.add_argument('--useRandomMatrix', type=str2bool, default=False, help='whether to use Random Matrix') # 这个是用来初始化GCN构造的那个矩阵的一种方法
    parser.add_argument('--useAllOneMatrix', type=str2bool, default=False, help='whether to use All One Matrix') # 这个也是用来初始化GCN构造的那个矩阵的一种方法

    parser.add_argument('--useCov', type=str2bool, default=False, help='whether to use Cov') #  一种画图的方法
    parser.add_argument('--useCluster', type=str2bool, default=True, help='whether to use Cluster') # 后面在话TSNE画图的时候用到的

    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=['resnet50', ])

    parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)') #  最后输出的分类的类别数目

    parser.add_argument('--num-class', default=7, type=int)
    parser.add_argument('--low-dim', default=64, type=int,
                        help='embedding dimension')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='momentum for updating momentum encoder')
    parser.add_argument('--proto_m', default=0.999, type=float,
                        help='momentum for computing the momving average of prototypes')
    parser.add_argument('--num_divided', type=int, default=10, help='the number of blocks [0, loge(7)] to be divided')

    parser.add_argument('--lr_ad', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')          #  随机种子


    return parser.parse_args()


def Train(args, model, train_source_dataloader, train_target_dataloader, labeled_train_target_dataloader, optimizer, epoch, writer):
    """Train."""

    model.train()
    # torch.autograd.set_detect_anomaly(True) # 正向传播的时候开启求导异常的检测

    #! 注释
    # acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)] # 每个变量都装载了7个AverageMeter对象，AverageMeter对象存储变量易于更新和修改。
    # loss, global_cls_loss, local_cls_loss, afn_loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    acc, prec, recall = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for
                                                                                 i in range(7)], [
                            [AverageMeter() for i in range(7)] for i in range(7)]
    cls_loss, dan_loss, loss, data_time, batch_time = [AverageMeter() for i in range(7)], [AverageMeter() for i in
                                                                                           range(
                                                                                               7)], AverageMeter(), AverageMeter(), AverageMeter()
    data_time, batch_time = AverageMeter(), AverageMeter()
    num_ADNets = [0 for i in range(7)]
    six_probilities_source, six_accuracys_source = [AverageMeter() for i in range(6)], [AverageMeter() for i in
                                                                                        range(6)]
    six_probilities_target, six_accuracys_target = [AverageMeter() for i in range(6)], [AverageMeter() for i in
                                                                                        range(6)]

    # @ 特别记录一下train中的target
    acc_target, prec_target, recall_target = [[AverageMeter() for i in range(7)] for i in range(7)], [
        [AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]

    # @ 在七个分类器的预测值都相同的情况下再增加多一个entropy的condition（这些统计参数只用于target domain）
    # delta = 1.9459 / args.num_divided / args.num_divided
    delta = 0.19459 / args.num_divided
    entropy_thresholds = np.arange(delta, 0.19459 + delta, delta)
    probilities_entropy, accuracys_entropy = [AverageMeter() for i in range(args.num_divided)], [AverageMeter() for i in
                                                                                                 range(
                                                                                                     args.num_divided)]

    source_cls_loss, target_cls_loss = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]
    # Decay Learn Rate per Epoch，不同的主干采用不同的学习率迭代
    if args.Backbone in ['ResNet18', 'ResNet50']:
        if epoch <= 10:
            args.lr = 1e-4
        elif epoch <= 40:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    elif args.Backbone == 'MobileNet':
        if epoch <= 20:
            args.lr = 1e-3
        elif epoch <= 40:
            args.lr = 1e-4
        elif epoch <= 60:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    elif args.Backbone == 'VGGNet':
        if epoch <= 30:
            args.lr = 1e-3
        elif epoch <= 60:
            args.lr = 1e-4
        elif epoch <= 70:
            args.lr = 1e-5
        else:
            args.lr = 1e-6

    iter_source_dataloader = iter(train_source_dataloader)
    iter_target_dataloader = iter(train_target_dataloader)
    if labeled_train_target_dataloader != None:
        iter_labeled_target_dataloader = iter(labeled_train_target_dataloader)
    num_iter = len(train_source_dataloader) if (len(train_source_dataloader) > len(train_target_dataloader)) else len(
        train_target_dataloader)
    end = time.time()
    train_bar = tqdm(range(num_iter))
    for step, batch_index in enumerate(train_bar):
        try:
            _, data_source, landmark_source, label_source = iter_source_dataloader.next()
        except:
            iter_source_dataloader = iter(train_source_dataloader)
            _, data_source, landmark_source, label_source = iter_source_dataloader.next()
        try:
            _, data_target, landmark_target, label_target = iter_target_dataloader.next()
        except:
            iter_target_dataloader = iter(train_target_dataloader)
            _, data_target, landmark_target, label_target = iter_target_dataloader.next()

        data_time.update(time.time() - end)

        data_source, landmark_source, label_source = data_source.cuda(), landmark_source.cuda(), label_source.cuda()
        data_target, landmark_target, label_target = data_target.cuda(), landmark_target.cuda(), label_target.cuda()

        # Forward Propagation
        end = time.time()
        # ! m21-11-13, 注释
        # feature, output, loc_output = model(torch.cat((data_source, data_target), 0), torch.cat((landmark_source, landmark_target), 0))
        features, preds, _ = model(torch.cat((data_source, data_target), 0),
                                torch.cat((landmark_source, landmark_target), 0),
                                torch.cat((label_source, label_target), 0),
                                args)
        batch_time.update(time.time() - end)

        # Compute Loss
        '''
            global_cls_loss_ = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source) # 这里计算交叉熵的时候都是只用了Source Domain的部分和Label
            local_cls_loss_ = nn.CrossEntropyLoss()(loc_output.narrow(0, 0, data_source.size(0)), label_source) if args.useLocalFeature else 0
            afn_loss_ = (HAFN(feature, args.weight_L2norm, args.radius) if args.methodOfAFN == 'HAFN' else SAFN(feature, args.weight_L2norm, args.deltaRadius)) if args.useAFN else 0
        '''

        # @ m21-11-13, Compute Classifier Loss(Source Domain)
        classifiers_loss_ratio = [7, 1, 1, 1, 1, 1, 7]
        loss_ = 0
        criteria = nn.CrossEntropyLoss()
        for i in range(7):
            tmp = criteria(preds[i].narrow(0, 0, data_source.size(0)), label_source)
            cls_loss[i].update(tmp.cpu().data.item(), data_source.size(0))
            source_cls_loss[i].update(tmp, data_source.size(0))
            loss_ += classifiers_loss_ratio[i] * tmp

        if labeled_train_target_dataloader != None:
            try:
                _, data_labeled_target, landmark_labeled_target, label_labeled_target = iter_labeled_target_dataloader.next()
            except:
                iter_labeled_target_dataloader = iter(labeled_train_target_dataloader)
                _, data_labeled_target, landmark_labeled_target, label_labeled_target = iter_labeled_target_dataloader.next()
            data_labeled_target, landmark_labeled_target, label_labeled_target = data_labeled_target.cuda(), landmark_labeled_target.cuda(), label_labeled_target.cuda()
            features_faked, preds_faked, _ = model(data_labeled_target, landmark_labeled_target, label_labeled_target, args)
            criteria = nn.CrossEntropyLoss()
            for i in range(7):
                tmp = criteria(preds_faked[i], label_labeled_target)
                cls_loss[i].update(tmp.cpu().data.item(), data_labeled_target.size(0))
                target_cls_loss[i].update(tmp, data_labeled_target.size(0))
                # loss_ += args.target_loss_ratio * classifiers_loss_ratio[i] * tmp
                loss_ += tmp
        # Back Propagation
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss_.backward()
        optimizer.step()

        # Decay Learn Rate
        optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr, weight_decay=args.weight_decay) # optimizer = lr_scheduler(optimizer, num_iter*(epoch-1)+step, 0.001, 0.75, lr=args.lr, weight_decay=args.weight_decay)
        # Compute accuracy, recall and loss
        for classifier_id in range(7):  # 遍历七个判别器，并计算对应的metrics
            Compute_Accuracy(args, preds[classifier_id].narrow(0, 0, data_source.size(0)), label_source,
                             acc[classifier_id], prec[classifier_id], recall[classifier_id])

        # Compute accuracy, precision and recall
        for classifier_id in range(7):  # 遍历七个判别器，并计算对应的metrics
            Compute_Accuracy(args, preds[classifier_id].narrow(0, data_target.size(0), data_target.size(0)),
                             label_target, acc_target[classifier_id], prec_target[classifier_id],
                             recall_target[classifier_id])

        Count_Probility_Accuracy(six_probilities_source, six_accuracys_source,
                                 [pred.narrow(0, 0, data_source.size(0)) for pred in preds], label_source)
        Count_Probility_Accuracy(six_probilities_target, six_accuracys_target,
                                 [pred.narrow(0, data_target.size(0), data_target.size(0)) for pred in preds],
                                 label_target)

        Count_Probility_Accuracy_Entropy(entropy_thresholds, probilities_entropy, accuracys_entropy,
                                         [pred.narrow(0, data_target.size(0), data_target.size(0)) for pred in preds],
                                         label_target)

        # Log loss
        loss.update(float(loss_.cpu().data.item()), data_source.size(0))

        end = time.time()
        train_bar.desc = f'[Train (Source Domain) Epoch {epoch}/{args.epochs}] cls loss: {cls_loss[0].avg:.3f}, {cls_loss[1].avg:.3f}, {cls_loss[2].avg:.3f}, {cls_loss[3].avg:.3f}, {cls_loss[4].avg:.3f}, {cls_loss[5].avg:.3f}, {cls_loss[6].avg:.3f}'
    
    #! m21-11-12, 注释
    # AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    #@ m21-11-12, 只记录七个分类器的准确率
    accs = Show_OnlyAccuracy(acc)

    accs_target = Show_OnlyAccuracy(acc_target)

    writer.add_scalars('Train/CLS_Acc_Source',
                       {'global': accs[0], 'left_eye': accs[1], 'right_eye': accs[2], 'nose': accs[3],
                        'left_mouse': accs[4], 'right_mouse': accs[5], 'global_local': accs[6]}, epoch)
    writer.add_scalars('Train/CLS_Loss_Source',
                       {'global': cls_loss[0].avg, 'left_eye': cls_loss[1].avg, 'right_eye': cls_loss[2].avg,
                        'nose': cls_loss[3].avg, 'left_mouse': cls_loss[4].avg, 'right_mouse': cls_loss[5].avg,
                        'global_local': cls_loss[6].avg}, epoch)
    # writer.add_scalars('Train/Six_Probilities_Source', {'Situation_0': six_probilities_source[0].avg, 'Situation_1':six_probilities_source[1].avg, 'Situation_2':six_probilities_source[2].avg, 'Situation_3':six_probilities_source[3].avg, 'Situation_4':six_probilities_source[4].avg, 'Situation_5':six_probilities_source[5].avg}, epoch)
    # writer.add_scalars('Train/Six_Accuracys_Source', {'Situation_0': six_accuracys_source[0].avg, 'Situation_1':six_accuracys_source[1].avg, 'Situation_2':six_accuracys_source[2].avg, 'Situation_3':six_accuracys_source[3].avg, 'Situation_4':six_accuracys_source[4].avg, 'Situation_5':six_accuracys_source[5].avg}, epoch)
    # writer.add_scalars('Train/Six_Probilities_Target', {'Situation_0': six_probilities_target[0].avg, 'Situation_1':six_probilities_target[1].avg, 'Situation_2':six_probilities_target[2].avg, 'Situation_3':six_probilities_target[3].avg, 'Situation_4':six_probilities_target[4].avg, 'Situation_5':six_probilities_target[5].avg}, epoch)
    # writer.add_scalars('Train/Six_Accuracys_Target', {'Situation_0': six_accuracys_target[0].avg, 'Situation_1':six_accuracys_target[1].avg, 'Situation_2':six_accuracys_target[2].avg, 'Situation_3':six_accuracys_target[3].avg, 'Situation_4':six_accuracys_target[4].avg, 'Situation_5':six_accuracys_target[5].avg}, epoch)

    writer.add_scalars('Train/Merely_Source_CLS_Loss',
                       {'global': source_cls_loss[0].avg, 'left_eye': source_cls_loss[1].avg,
                        'right_eye': source_cls_loss[2].avg, 'nose': source_cls_loss[3].avg,
                        'left_mouse': source_cls_loss[4].avg, 'right_mouse': source_cls_loss[5].avg,
                        'global_local': source_cls_loss[6].avg}, epoch)
    writer.add_scalars('Train/Merely_Target_CLS_Loss',
                       {'global': target_cls_loss[0].avg, 'left_eye': target_cls_loss[1].avg,
                        'right_eye': target_cls_loss[2].avg, 'nose': target_cls_loss[3].avg,
                        'left_mouse': target_cls_loss[4].avg, 'right_mouse': target_cls_loss[5].avg,
                        'global_local': target_cls_loss[6].avg}, epoch)

    writer.add_scalars('Train/CLS_Acc_Target',
                       {'global': accs_target[0], 'left_eye': accs_target[1], 'right_eye': accs_target[2],
                        'nose': accs_target[3], 'left_mouse': accs_target[4], 'right_mouse': accs_target[5],
                        'global_local': accs_target[6]}, epoch)
    acc_dic, pro_dic = {}, {}
    for i in range(args.num_divided):
        acc_dic.update({'entropy_' + str(entropy_thresholds[i]): accuracys_entropy[i].avg})
        pro_dic.update({'entropy_' + str(entropy_thresholds[i]): probilities_entropy[i].avg})
    writer.add_scalars('Train/Accuracys_Entropy', acc_dic, epoch)
    writer.add_scalars('Train/Probility_Entropy', pro_dic, epoch)
    dan_accs = 0

    # LoggerInfo = '[Train Epoch {0}]： Learning Rate {1}  DAN Learning Rate {2}\n' \
    #              'CLS_Acc:  global {cls_accs[0]:.4f}\t left_eye {cls_accs[1]:.4f}\t right_eye {cls_accs[2]:.4f}\t nose {cls_accs[3]:.4f}\t left_mouse {cls_accs[4]:.4f}\t right_mouse {cls_accs[5]:.4f}\t global_local {cls_accs[6]:.4f}\n' \
    #              'DAN_ACC:  global {dan_accs[0]:.4f}\t left_eye {dan_accs[1]:.4f}\t right_eye {dan_accs[2]:.4f}\t nose {dan_accs[3]:.4f}\t left_mouse {dan_accs[4]:.4f}\t right_mouse {dan_accs[5]:.4f}\t global_local {dan_accs[6]:.4f}\n' \
    #              'SUM_Loss: global {cls_loss[0].avg:.4f}\t left_eye {cls_loss[1].avg:.4f}\t right_eye {cls_loss[2].avg:.4f}\t nose {cls_loss[3].avg:.4f}\t left_mouse {cls_loss[4].avg:.4f}\t right_mouse {cls_loss[5].avg:.4f}\t global_local {cls_loss[6].avg:.4f}\n' \
    #              'DAN_Loss: global {dan_loss[0].avg:.4f}\t left_eye {dan_loss[1].avg:.4f}\t right_eye {dan_loss[2].avg:.4f}\t nose {dan_loss[3].avg:.4f}\t left_mouse {dan_loss[4].avg:.4f}\t right_mouse {dan_loss[5].avg:.4f}\t global_local {dan_loss[6].avg:.4f}\n' \
    #              'Situ_Acc_Source: Situation_0 {six_acc_source[0].avg:.4f}\t Situation_1 {six_acc_source[1].avg:.4f}\t Situation_2 {six_acc_source[2].avg:.4f}\t Situation_3 {six_acc_source[3].avg:.4f}\t Situation_4 {six_acc_source[4].avg:.4f}\t Situation_5 {six_acc_source[5].avg:.4f}\n' \
    #              'Situ_Pro_Source: Situation_0 {six_prob_source[0].avg:.4f}\t Situation_1 {six_prob_source[1].avg:.4f}\t Situation_2 {six_prob_source[2].avg:.4f}\t Situation_3 {six_prob_source[3].avg:.4f}\t Situation_4 {six_prob_source[4].avg:.4f}\t Situation_5 {six_prob_source[5].avg:.4f}\n' \
    #              'Situ_Acc_Target: Situation_0 {six_acc_target[0].avg:.4f}\t Situation_1 {six_acc_target[1].avg:.4f}\t Situation_2 {six_acc_target[2].avg:.4f}\t Situation_3 {six_acc_target[3].avg:.4f}\t Situation_4 {six_acc_target[4].avg:.4f}\t Situation_5 {six_acc_target[5].avg:.4f}\n' \
    #              'Situ_Pro_Target: Situation_0 {six_prob_target[0].avg:.4f}\t Situation_1 {six_prob_target[1].avg:.4f}\t Situation_2 {six_prob_target[2].avg:.4f}\t Situation_3 {six_prob_target[3].avg:.4f}\t Situation_4 {six_prob_target[4].avg:.4f}\t Situation_5 {six_prob_target[5].avg:.4f}\n' \
    #              'Source_CLS_Loss: global {source_cls_loss[0].avg:.4f}\t left_eye {source_cls_loss[1].avg:.4f}\t right_eye {source_cls_loss[2].avg:.4f}\t nose {source_cls_loss[3].avg:.4f}\t left_mouse {source_cls_loss[4].avg:.4f}\t right_mouse {source_cls_loss[5].avg:.4f}\t global_local {source_cls_loss[6].avg:.4f}\n' \
    #              'Target_CLS_Loss: global {target_cls_loss[0].avg:.4f}\t left_eye {target_cls_loss[1].avg:.4f}\t right_eye {target_cls_loss[2].avg:.4f}\t nose {target_cls_loss[3].avg:.4f}\t left_mouse {target_cls_loss[4].avg:.4f}\t right_mouse {target_cls_loss[5].avg:.4f}\t global_local {target_cls_loss[6].avg:.4f}\n\n' \
    #     .format(epoch, args.lr, args.lr_ad, cls_accs=accs, dan_accs=dan_accs, cls_loss=cls_loss, dan_loss=dan_loss,
    #             six_acc_source=six_accuracys_source, six_prob_source=six_probilities_source,
    #             six_acc_target=six_accuracys_target, six_prob_target=six_probilities_target,
    #             source_cls_loss=source_cls_loss, target_cls_loss=target_cls_loss)

    # with open(args.OutputPath + "/train_result_transfer.log", "a") as f:
    #     f.writelines(LoggerInfo)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def Test(args, model, test_source_dataloader, test_target_dataloader, Best_Acc, epoch, confidence_target, writer, optimizer):
    """Test."""

    model.eval()
    # torch.autograd.set_detect_anomaly(True)

    #! m21-11-12
    # iter_source_dataloader = iter(test_source_dataloader)
    # iter_target_dataloader = iter(test_target_dataloader)

    # Test on Source Domain
    acc, prec, recall = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    cls_loss, loss, afn_loss, data_time, batch_time = [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    six_probilities, six_accuracys = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]
    
    end = time.time()
    test_source_bar = tqdm(test_source_dataloader)
    for step, (_, input, landmark, label) in enumerate(test_source_bar):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        data_time.update(time.time() - end)
        
        # Forward Propagation
        with torch.no_grad():
            end = time.time()
            #! m21-11-12，注释
            # feature, output, loc_output = model(input, landmark)
            #@ m21-11-11, 新的model调用
            features, preds, target = model(input, landmark, label, args, is_eval=True)

            batch_time.update(time.time()-end)
        


        #@ m21-11-11, 新的计算loss的方式
        loss_ = 0
        criteria = nn.CrossEntropyLoss()
        for i in range(7):
            tmp = criteria(preds[i], label)
            cls_loss[i].update(tmp.cpu().data.item(), input.size(0))
            loss_ += tmp

        # Compute accuracy, precision and recall
        for classifier_id in range(7): # 遍历七个判别器，并计算对应的metrics
            Compute_Accuracy(args, preds[classifier_id], label, acc[classifier_id], prec[classifier_id], recall[classifier_id])

        Count_Probility_Accuracy(six_probilities, six_accuracys, preds, label)

        # Log loss
        loss.update(float(loss_.cpu().data.item()), input.size(0))
        end = time.time()
        test_source_bar.desc = f'[Test (Source Domain) Epoch {epoch}/{args.epochs}] cls loss: {cls_loss[0].avg:.3f}, {cls_loss[1].avg:.3f}, {cls_loss[2].avg:.3f}, {cls_loss[3].avg:.3f}, {cls_loss[4].avg:.3f}, {cls_loss[5].avg:.3f}, {cls_loss[6].avg:.3f}'

    #! m21-11-12, 注释
    # AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    #@ m21-11-12, 只记录七个分类器的准确率
    accs = Show_OnlyAccuracy(acc)
    writer.add_scalars('Accuracy/Test_Source', {'global': accs[0], 'left_eye':accs[1], 'right_eye':accs[2], 'nose':accs[3], 'left_mouse':accs[4], 'right_mouse':accs[5], 'global_local':accs[6]}, epoch)
    writer.add_scalars('Loss/Test_Source', {'global': cls_loss[0].avg, 'left_eye':cls_loss[1].avg, 'right_eye':cls_loss[2].avg, 'nose':cls_loss[3].avg, 'left_mouse':cls_loss[4].avg, 'right_mouse':cls_loss[5].avg, 'global_local':cls_loss[6].avg}, epoch)
    writer.add_scalars('Six_Probilities/Test_Source', {'situation_0': six_probilities[0].avg, 'situation_1':six_probilities[1].avg, 'situation_2':six_probilities[2].avg, 'situation_3':six_probilities[3].avg, 'situation_4':six_probilities[4].avg, 'situation_5':six_probilities[5].avg}, epoch)
    writer.add_scalars('Six_Accuracys/Test_Source', {'situation_0': six_accuracys[0].avg, 'situation_1':six_accuracys[1].avg, 'situation_2':six_accuracys[2].avg, 'situation_3':six_accuracys[3].avg, 'situation_4':six_accuracys[4].avg, 'situation_5':six_accuracys[5].avg}, epoch)

    LoggerInfo = '[Test (Source Domain) Epoch {0}]： Learning Rate {1} \n'\
    'Accuracy: global {Accuracys[0]:.4f}\t left_eye {Accuracys[1]:.4f}\t right_eye {Accuracys[2]:.4f}\t nose {Accuracys[3]:.4f}\t left_mouse {Accuracys[4]:.4f}\t right_mouse {Accuracys[5]:.4f}\t global_local {Accuracys[6]:.4f}\n'\
    'Cls_Loss: global {Loss[0].avg:.4f}\t left_eye {Loss[1].avg:.4f}\t right_eye {Loss[2].avg:.4f}\t nose {Loss[3].avg:.4f}\t left_mouse {Loss[4].avg:.4f}\t right_mouse {Loss[5].avg:.4f}\t global_local {Loss[6].avg:.4f}\n'\
    'Situ_Acc: Situation_0 {six_acc[0].avg:.4f}\t Situation_1 {six_acc[1].avg:.4f}\t Situation_2 {six_acc[2].avg:.4f}\t Situation_3 {six_acc[3].avg:.4f}\t Situation_4 {six_acc[4].avg:.4f}\t Situation_5 {six_acc[5].avg:.4f}\n'\
    'Situ_Pro: Situation_0 {six_prob[0].avg:.4f}\t Situation_1 {six_prob[1].avg:.4f}\t Situation_2 {six_prob[2].avg:.4f}\t Situation_3 {six_prob[3].avg:.4f}\t Situation_4 {six_prob[4].avg:.4f}\t Situation_5 {six_prob[5].avg:.4f}\n\n'\
    .format(epoch, args.lr, Accuracys=accs, Loss=cls_loss, six_acc=six_accuracys, six_prob=six_probilities)

    with open(args.OutputPath + "/test_result_source.log","a") as f:
        f.writelines(LoggerInfo)

    #@ Save Checkpoints
    classifier_name = {0:'global', 1:'left_eye', 2:'right_eye', 3:'nose', 4:'left_mouth', 5:'right_mouth', 6:'global_local'}
    best_classifier_id = accs.index(max(accs))
    best_classifier = classifier_name[best_classifier_id]
    best_acc = accs[best_classifier_id]
    if best_acc > Best_Acc and (confidence_target > 0.965 or epoch < 10):     # 根据Source Domain的效果判断是否存储
        Best_Acc = best_acc
        print("**************")
        print(f'[Save] Best Acc: {Best_Acc:.4f}, the classifier is {best_classifier}. Save the checkpoint! （Target Confidence is {confidence_target}）')
        print("**************")

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))
        else:
            torch.save(model.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))
            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': args.arch,
            #     'state_dict': model.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            # }, is_best=False, filename='{}/checkpoint_{:04d}.pth.tar'.format(args.OutputPath, epoch))

    #@ ===========================================================================================
    
    # Test on Target Domain
    acc, prec, recall = [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)], [[AverageMeter() for i in range(7)] for i in range(7)]
    cls_loss, loss, afn_loss, data_time, batch_time = [AverageMeter() for i in range(7)], AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    six_probilities, six_accuracys = [AverageMeter() for i in range(6)], [AverageMeter() for i in range(6)]

    end = time.time()
    test_target_bar = tqdm(test_target_dataloader)
    for step, (_, input, landmark, label) in enumerate(test_target_bar):
        input, landmark, label = input.cuda(), landmark.cuda(), label.cuda()
        data_time.update(time.time() - end)
        
        # Forward Propagation
        with torch.no_grad():
            end = time.time()
            #! m21-11-12，注释
            # feature, output, loc_output = model(input, landmark)
            #@ m21-11-11, 新的model调用
            features, preds, target = model(input, landmark, label, args, is_eval=True)
            batch_time.update(time.time()-end)
        
        # Compute Loss
        #! m21-11-11, 注释
        '''
            global_cls_loss_ = nn.CrossEntropyLoss()(output, label)
            local_cls_loss_ = nn.CrossEntropyLoss()(loc_output, label) if args.useLocalFeature else 0
            afn_loss_ = (HAFN(feature, args.weight_L2norm, args.radius) if args.methodOfAFN == 'HAFN' else SAFN(feature, args.weight_L2norm, args.deltaRadius)) if args.useAFN else 0
            loss_ = global_cls_loss_ + local_cls_loss_ + (afn_loss_ if args.useAFN else 0)
        '''

        #@ m21-11-12, 新的计算loss的方式
        loss_ = 0
        criteria = nn.CrossEntropyLoss()
        for i in range(7):
            tmp = criteria(preds[i], label)
            cls_loss[i].update(tmp.cpu().data.item(), input.size(0))
            loss_ += tmp

        # Compute accuracy, precision and recall
        for classifier_id in range(7): # 遍历七个判别器，并计算对应的metrics
            Compute_Accuracy(args, preds[classifier_id], label, acc[classifier_id], prec[classifier_id], recall[classifier_id])

        Count_Probility_Accuracy(six_probilities, six_accuracys, preds, label)

        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        end = time.time()
        test_target_bar.desc = f'[Test (Target Domain) Epoch {epoch}/{args.epochs}] cls loss: {cls_loss[0].avg:.3f}, {cls_loss[1].avg:.3f}, {cls_loss[2].avg:.3f}, {cls_loss[3].avg:.3f}, {cls_loss[4].avg:.3f}, {cls_loss[5].avg:.3f}, {cls_loss[6].avg:.3f}'

    #! m21-11-12, 注释
    # AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)
    #@ m21-11-12, 只记录七个分类器的准确率
    accs = Show_OnlyAccuracy(acc)
    writer.add_scalars('Accuracy/Test_Target', {'global': accs[0], 'left_eye':accs[1], 'right_eye':accs[2], 'nose':accs[3], 'left_mouse':accs[4], 'right_mouse':accs[5], 'global_local':accs[6]}, epoch)
    writer.add_scalars('Loss/Test_Target', {'global': cls_loss[0].avg, 'left_eye':cls_loss[1].avg, 'right_eye':cls_loss[2].avg, 'nose':cls_loss[3].avg, 'left_mouse':cls_loss[4].avg, 'right_mouse':cls_loss[5].avg, 'global_local':cls_loss[6].avg}, epoch)
    writer.add_scalars('Six_Probilities/Test_Target', {'situation_0': six_probilities[0].avg, 'situation_1':six_probilities[1].avg, 'situation_2':six_probilities[2].avg, 'situation_3':six_probilities[3].avg, 'situation_4':six_probilities[4].avg, 'situation_5':six_probilities[5].avg}, epoch)
    writer.add_scalars('Six_Accuracys/Test_Target', {'situation_0': six_accuracys[0].avg, 'situation_1':six_accuracys[1].avg, 'situation_2':six_accuracys[2].avg, 'situation_3':six_accuracys[3].avg, 'situation_4':six_accuracys[4].avg, 'situation_5':six_accuracys[5].avg}, epoch)

    LoggerInfo = '[Test (Target Domain) Epoch {0}]： Learning Rate {1} \n'\
    'Accuracy: global {Accuracys[0]:.4f}\t left_eye {Accuracys[1]:.4f}\t right_eye {Accuracys[2]:.4f}\t nose {Accuracys[3]:.4f}\t left_mouse {Accuracys[4]:.4f}\t right_mouse {Accuracys[5]:.4f}\t global_local {Accuracys[6]:.4f}\n'\
    'Cls_Loss: global {Loss[0].avg:.4f}\t left_eye {Loss[1].avg:.4f}\t right_eye {Loss[2].avg:.4f}\t nose {Loss[3].avg:.4f}\t left_mouse {Loss[4].avg:.4f}\t right_mouse {Loss[5].avg:.4f}\t global_local {Loss[6].avg:.4f}\n'\
    'Situ_Acc: Situation_0 {six_acc[0].avg:.4f}\t Situation_1 {six_acc[1].avg:.4f}\t Situation_2 {six_acc[2].avg:.4f}\t Situation_3 {six_acc[3].avg:.4f}\t Situation_4 {six_acc[4].avg:.4f}\t Situation_5 {six_acc[5].avg:.4f}\n'\
    'Situ_Pro: Situation_0 {six_prob[0].avg:.4f}\t Situation_1 {six_prob[1].avg:.4f}\t Situation_2 {six_prob[2].avg:.4f}\t Situation_3 {six_prob[3].avg:.4f}\t Situation_4 {six_prob[4].avg:.4f}\t Situation_5 {six_prob[5].avg:.4f}\n\n'\
    .format(epoch, args.lr, Accuracys=accs, Loss=cls_loss, six_acc=six_accuracys, six_prob=six_probilities)

    with open(args.OutputPath + "/test_result_target.log","a") as f:
        f.writelines(LoggerInfo)

    return Best_Acc

def main():
    """Main."""
 
    # Parse Argument
    args = construct_args()         # 构造参数
    torch.manual_seed(args.seed)    # 人工种子
    folder = str(int(time.time()))
    print(f"Timestamp is {folder}")
    args.OutputPath = os.path.join(args.OutputPath, folder)
    makeFolder(args)

    # Experiment Information
    print('Log Name: %s' % args.Log_Name)
    print('Output Path: %s' % args.OutputPath)
    print('Backbone: %s' % args.Backbone)
    print('Resume Model: %s' % args.Resume_Model)
    print('CUDA_VISIBLE_DEVICES: %s' % args.GPU_ID)

    print('================================================')

    print('Use {} * {} Image'.format(args.faceScale, args.faceScale))
    print('SourceDataset: %s' % args.sourceDataset)
    print('TargetDataset: %s' % args.targetDataset)
    print('Train Batch Size: %d' % args.train_batch_size)
    print('Test Batch Size: %d' % args.test_batch_size)

    print('================================================')

    if args.showFeature:
        print('Show Visualiza Result of Feature.')

    if args.isTest:# 只是测试一下模型的性能
        print('Test Model.')
    else: # 正常的训练，打印训练参数
        print('Train Epoch: %d' % args.epochs)
        print('Learning Rate: %f' % args.lr)
        print('Momentum: %f' % args.momentum)
        print('Weight Decay: %f' % args.weight_decay)

        if args.useAFN: # AFN的方法
            print('Use AFN Loss: %s' % args.methodOfAFN)
            if args.methodOfAFN == 'HAFN':    # hard afn
                print('Radius of HAFN Loss: %f' % args.radius)
            else:                           # soft afn
                print('Delta Radius of SAFN Loss: %f' % args.deltaRadius)
            print('Weight L2 nrom of AFN Loss: %f' % args.weight_L2norm)

    print('================================================')

    print('Number of classes : %d' % args.class_num) # 表情的类别数
    if not args.useLocalFeature:
        print('Only use global feature.') # 只使用全局特征
    else:
        print('Use global feature and local feature.')

        if args.useIntraGCN:
            print('Use Intra GCN.') # 是否使用域内GCN进行传播
        if args.useInterGCN:
            print('Use Inter GCN.') # 是否使用域间GCN进行传播

        if args.useRandomMatrix and args.useAllOneMatrix:
            print('Wrong : Use RandomMatrix and AllOneMatrix both!')
            return None
        elif args.useRandomMatrix:
            print('Use Random Matrix in GCN.')
        elif args.useAllOneMatrix:
            print('Use All One Matrix in GCN.')

        if args.useCov and args.useCluster: # 使用协方差矩阵进行初始化or采用k-means算法进行初始化
            print('Wrong : Use Cov and Cluster both!')
            return None
        else:
            if args.useCov:
                print('Use Mean and Cov.') #todo: mean是指什么？
            else:
                print('Use Mean.') if not args.useCluster else print('Use Mean in Cluster.')

    print('================================================')

    print('================================================')
    # Bulid Model
    print('Building Model...')
    # model = BulidModel(args)
    model = MoPro(args)

    with open(args.OutputPath + "/first.log", "a") as f:
        num = 0
        for k, v in model.named_parameters():
            num += 1
            f.writelines(str(num) + "、" + str(k) + "\n" + str(v) + "\n\n")
    print('Done!')
    print('================================================')


    # Bulid Dataloder
    print("Building Train and Test Dataloader...")
    train_source_dataloader, _ = BulidDataloader(args, flag1='train', flag2='source') # 构建了source的源域数据集生成器。调用他之后能够返回batch size数量的（裁剪后的人脸图片，人脸的五个关键点，表情标签）
    train_target_dataloader, init_train_dataset_data = BulidDataloader(args, flag1='train', flag2='target') # 构建了训练的target目标域的数据集生成器，调用他之后能够返回batch size数量的（裁剪后的人脸图片，人脸的五个关键点，表情标签）
    test_source_dataloader, _ = BulidDataloader(args, flag1='test', flag2='source')   # test跟train的数据集生成器只有图像的预处理那里会有点不同
    test_target_dataloader, _ = BulidDataloader(args, flag1='test', flag2='target')

    print('Done!')

    #  Set Optimizer
    print('Building Optimizer...')
    model.cuda()
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    print('Done!')

    print('================================================')

    # Save Best Checkpoint
    Best_Acc = 0
    confidence = 0

    # Running Experiment
    print("Run Experiment...")
    writer = SummaryWriter(os.path.join(args.OutputPath, 'visual_board'))


    for epoch in range(1, args.epochs + 1):
        if args.showFeature and epoch % 5 == 1:
            print(f"=================\ndraw the tSNE graph...")
            Visualization(args.OutputPath + '/result_pics/train/source/{}_Source.jpg'.format(epoch), model, dataloader=train_source_dataloader, useClassify=True, domain='Source')
            Visualization(args.OutputPath + '/result_pics/train/target/{}_Target.jpg'.format(epoch), model, train_target_dataloader, useClassify=True, domain='Target')

            VisualizationForTwoDomain(args.OutputPath + '/result_pics/train_tow_domain/{}_train'.format(epoch), model, train_source_dataloader, train_target_dataloader, useClassify=True, showClusterCenter=False)
            VisualizationForTwoDomain(args.OutputPath + '/result_pics/test_tow_domain/{}_test'.format(epoch), model, test_source_dataloader, test_target_dataloader, useClassify=True, showClusterCenter=False)
            print(f"finish drawing!\n=================")

        if not args.isTest:
            if args.useCluster and epoch % 5 == 1:
                print(f"=================\nupdate the running_mean...")
                Initialize_Mean_Cluster(args, model, True, train_source_dataloader, train_target_dataloader)
                torch.cuda.empty_cache()
                print(f"finish the updating!\n=================")
            # Train(args, model, train_source_dataloader, optimizer, epoch, writer)
            
            if epoch >=10:
                train_fake_dataloader, confidence, probility = BuildLabeledDataloader(args, train_target_dataloader, init_train_dataset_data, model)
            else:
                train_fake_dataloader = None
            Train(args, model, train_source_dataloader, train_target_dataloader, train_fake_dataloader,optimizer, epoch, writer)
        Best_Acc = Test(args, model, test_source_dataloader, test_target_dataloader, Best_Acc, epoch, confidence, writer, optimizer)

        torch.cuda.empty_cache()

    writer.close()

if __name__ == '__main__':
    main()
