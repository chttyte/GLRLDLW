import os
import sys
import time
from collections import Counter
from copy import deepcopy

import tqdm
import shutil
import argparse
import subprocess
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
# from pytorch_lightning.distributed import dist
from torch.utils.tensorboard import SummaryWriter

from Loss import Entropy, DANN, CDAN, HAFN, SAFN
from model import *
from Utils import *


def construct_args():
    parser = argparse.ArgumentParser(description='Domain adaptation for Expression Classification')

    parser.add_argument('--Log_Name', type=str, help='Log Name')
    parser.add_argument('--OutputPath', type=str, help='Output Path')
    parser.add_argument('--Backbone', type=str, default='ResNet50',
                        choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
    parser.add_argument('--Resume_Model', type=str, help='Resume_Model', default='None')
    parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

    parser.add_argument('--useDAN', type=str2bool, default=False, help='whether to use DAN Loss')
    parser.add_argument('--methodOfDAN', type=str, default='CDAN-E', choices=['CDAN', 'CDAN-E', 'DANN'])

    parser.add_argument('--useAFN', type=str2bool, default=False, help='whether to use AFN Loss')
    parser.add_argument('--methodOfAFN', type=str, default='SAFN', choices=['HAFN', 'SAFN'])
    parser.add_argument('--radius', type=float, default=25.0, help='radius of HAFN (default: 25.0)')
    parser.add_argument('--deltaRadius', type=float, default=1.0, help='radius of SAFN (default: 1.0)')
    parser.add_argument('--weight_L2norm', type=float, default=0.05, help='weight L2 norm of AFN (default: 0.05)')

    parser.add_argument('--faceScale', type=int, default=112, help='Scale of face (default: 112)')
    parser.add_argument('--sourceDataset', type=str, default='RAF', choices=['RAF', 'AFED', 'MMI', 'FER2013'])
    parser.add_argument('--targetDataset', type=str, default='CK+',
                        choices=['RAF', 'CK+', 'JAFFE', 'MMI', 'Oulu-CASIA', 'SFEW', 'FER2013', 'ExpW', 'AFED', 'WFED'])
    parser.add_argument('--train_batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=64, help='input batch size for testing (default: 64)')
    parser.add_argument('--useMultiDatasets', type=str2bool, default=False, help='whether to use MultiDataset')

    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_ad', type=float, default=0.01)

    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='SGD weight decay (default: 0.0005)')

    parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model')
    parser.add_argument('--showFeature', type=str2bool, default=False, help='whether to show feature')

    parser.add_argument('--useIntraGCN', type=str2bool, default=False, help='whether to use Intra-GCN')
    parser.add_argument('--useInterGCN', type=str2bool, default=False, help='whether to use Inter-GCN')
    parser.add_argument('--useLocalFeature', type=str2bool, default=False, help='whether to use Local Feature')
    parser.add_argument('--useRandomMatrix', type=str2bool, default=False, help='whether to use Random Matrix')
    parser.add_argument('--useAllOneMatrix', type=str2bool, default=False, help='whether to use All One Matrix')
    parser.add_argument('--useCov', type=str2bool, default=False, help='whether to use Cov')
    parser.add_argument('--useCluster', type=str2bool, default=False, help='whether to use Cluster')

    parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num_divided', type=int, default=10, help='the number of blocks [0, loge(7)] to be divided')
    parser.add_argument('--randomLayer', type=str2bool, default=False, help='whether to use random')
    parser.add_argument('--thresh_warmup', type=str2bool, default=True, help='whether to warmup')

    parser.add_argument('--target_loss_ratio', type=int, default=5,
                        help='the ratio of seven classifier using on target label on the base of classifier_loss_ratio')

    parser.add_argument('--low-dim', default=64, type=int,
                        help='embedding dimension')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--moco_m', default=0.999, type=float,
                        help='momentum for updating momentum encoder')
    parser.add_argument('--proto_m', default=0.999, type=float,
                        help='momentum for computing the momving average of prototypes')
    parser.add_argument('--temperature', default=0.1, type=float,
                        help='contrastive temperature')
    parser.add_argument('--alpha', default=0.5, type=float,
                        help='weight to combine model prediction and prototype prediction')
    parser.add_argument('--pseudo_th', default=0.8, type=float,
                        help='threshold for pseudo labels')

    parser.add_argument('--moco_queue', default=8192, type=int,
                        help='queue size; number of negative samples')

    args = parser.parse_args()  # 构造参数
    return args


def Train(args, model, train_source_dataloader, train_target_dataloader, labeled_train_target_dataloader, optimizer,
          epoch, writer, labeled_dset_len):
    """Train."""
    model.train()
    # torch.autograd.set_detect_anomaly(True)

    acc_cls = AverageMeter2('Acc@Cls', ':2.2f')
    acc_proto = AverageMeter2('Acc@Proto', ':2.2f')
    acc_inst = AverageMeter2('Acc@Inst', ':2.2f')

    data_time, batch_time = AverageMeter(), AverageMeter()

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

    select_label_1 = torch.ones((labeled_dset_len,), dtype=torch.long) * -1
    select_label_1 = select_label_1.cuda()
    select_label_2 = torch.ones((labeled_dset_len,), dtype=torch.long) * -1
    select_label_2 = select_label_2.cuda()
    classwise_acc_1 = torch.zeros((args.class_num,)).cuda()
    classwise_acc_2 = torch.zeros((args.class_num,)).cuda()

    classwise_acc = []
    classwise_acc.append(classwise_acc_1)
    classwise_acc.append(classwise_acc_2)

    select_label = []
    select_label.append(select_label_1)
    select_label.append(select_label_2)


    # @ 统计一下在不同domain上训练的loss
    source_cls_loss, target_cls_loss = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]

    if args.useDAN:
        num_ADNet = 0

    # Decay Learn Rate per Epoch
    if epoch <= 30:
        args.lr, args.lr_ad = 1e-5, 0.0001
    elif epoch <= 50:
        args.lr, args.lr_ad = 5e-6, 0.0001
    else:
        args.lr, args.lr_ad = 2.5e-6, 0.00001

    # if epoch <= 30:
    #     args.lr, args.lr_ad = 5e-6, 0.0001
    # elif epoch <= 50:
    #     args.lr, args.lr_ad = 1e-6, 0.0001
    # else:
    #     args.lr, args.lr_ad = 1e-7, 0.00001

    optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr)

    # Get Source/Target Dataloader iterator
    iter_source_dataloader = iter(train_source_dataloader)
    iter_target_dataloader = iter(train_target_dataloader)
    if labeled_train_target_dataloader != None:
        iter_labeled_target_dataloader = iter(labeled_train_target_dataloader)

    # len(data_loader) = math.ceil(len(data_loader.dataset)/batch_size)
    num_iter = len(train_source_dataloader) if (len(train_source_dataloader) > len(train_target_dataloader)) else len(
        train_target_dataloader)

    end = time.time()
    train_bar = tqdm(range(num_iter))


    for step, batch_index in enumerate(train_bar):
        try:
            _, data_source, landmark_source, label_source, img_aug_src = next(iter_source_dataloader)
        except:
            iter_source_dataloader = iter(train_source_dataloader)
            _, data_source, landmark_source, label_source, img_aug_src = next(iter_source_dataloader)
        try:
            _, data_target, landmark_target, label_target, img_aug_tag = next(iter_target_dataloader)
        except:
            iter_target_dataloader = iter(train_target_dataloader)
            _, data_target, landmark_target, label_target, img_aug_tag = next(iter_target_dataloader)

        data_time.update(time.time() - end)

        data_source, landmark_source, label_source, img_aug_src = data_source.cuda(), landmark_source.cuda(), label_source.cuda(), img_aug_src.cuda()
        # data_target, landmark_target, label_target, img_aug_tag = data_target.cuda(), landmark_target.cuda(), label_target.cuda(), img_aug_tag.cuda()

        # Forward Propagation
        end = time.time()

        # feature, output, loc_output = model(torch.cat((data_source, data_target), 0), torch.cat((landmark_source, landmark_target), 0))
        features, preds = model(
            data_source,
            landmark_source,
            label_source,
            img_aug_src,
            args,
            classwise_acc,
            is_src=True)

        batch_time.update(time.time() - end)

        # Compute Loss
        '''
            global_cls_loss_ = nn.CrossEntropyLoss()(output.narrow(0, 0, data_source.size(0)), label_source) # 这里计算交叉熵的时候都是只用了Source Domain的部分和Label
            local_cls_loss_ = nn.CrossEntropyLoss()(loc_output.narrow(0, 0, data_source.size(0)), label_source) if args.useLocalFeature else 0
            afn_loss_ = (HAFN(feature, args.weight_L2norm, args.radius) if args.methodOfAFN == 'HAFN' else SAFN(feature, args.weight_L2norm, args.deltaRadius)) if args.useAFN else 0
        '''


        loss_ = 0
        criteria = nn.CrossEntropyLoss()
        classifiers_ratio_source = [1, 1, 1, 1, 1, 1, 1]


        for i in range(7):
            tmp = criteria(preds[i], label_source)
            loss_ += classifiers_ratio_source[i] * tmp


        # @ using the fake label of target domain to train the classifier
        if labeled_train_target_dataloader != None:

            try:
                labeled_idx, data_labeled_target, landmark_labeled_target, label_labeled_target, label_true, label_img_aug = next(iter_labeled_target_dataloader)
            except:
                iter_labeled_target_dataloader = iter(labeled_train_target_dataloader)
                labeled_idx, data_labeled_target, landmark_labeled_target, label_labeled_target, label_true, label_img_aug = next(iter_labeled_target_dataloader)

            data_labeled_target, landmark_labeled_target, label_labeled_target, label_true = \
                data_labeled_target.cuda(), landmark_labeled_target.cuda(), label_labeled_target.cuda(), label_true.cuda()

            labeled_idx = labeled_idx.cuda()
            p_cutoff = 0.98

            pseudo_counter_1 = Counter(select_label[0].tolist())
            pseudo_counter_2 = Counter(select_label[1].tolist())

            if max(pseudo_counter_1.values()) < labeled_dset_len and max(pseudo_counter_2.values()) < labeled_dset_len:
                if args.thresh_warmup:
                    for i in range(args.class_num):
                        classwise_acc[0][i] = pseudo_counter_1[i] / max(pseudo_counter_1.values())
                        classwise_acc[1][i] = pseudo_counter_2[i] / max(pseudo_counter_2.values())
                else:
                    wo_negative_one_1 = deepcopy(pseudo_counter_1)
                    wo_negative_one_2 = deepcopy(pseudo_counter_2)
                    if -1 in wo_negative_one_1.keys():
                        wo_negative_one_1.pop(-1)
                    if -1 in wo_negative_one_2.keys():
                        wo_negative_one_2.pop(-1)
                    for i in range(args.class_num):
                        classwise_acc[0][i] = pseudo_counter_1[i] / max(wo_negative_one_1.values())
                        classwise_acc[1][i] = pseudo_counter_2[i] / max(wo_negative_one_2.values())

            unsup_loss, unsup_loss_neg, select, pseudo_lb  = model(data_labeled_target,
                                                            landmark_labeled_target,
                                                            label_labeled_target,
                                                            label_img_aug,
                                                            args,
                                                            classwise_acc,
                                                            is_clean=True,
                                                            p_cutoff=p_cutoff
                                                            )
            # if step % 20 == 1:
            #     model.show_dy_threshold(args, classwise_acc)
            # if step % 10 == 1:
            #     for i in range(7):
            #         print(1 - torch.pow(classwise_acc[0][i], 2))

            if labeled_idx[select[0] == 1].nelement() != 0:
                select_label[0][labeled_idx[select[0] == 1]] = pseudo_lb[0][select[0] == 1]

            if labeled_idx[select[1] == 1].nelement() != 0:
                select_label[1][labeled_idx[select[1] == 1]] = pseudo_lb[1][select[1] == 1]



            #JAFFE loss_ration = 1.0
            # loss_ration = [7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 7.0]
            loss_ratio = 1.0


            # positive learning
            loss_ += loss_ratio * unsup_loss[0]
            loss_ += loss_ratio * unsup_loss[1]

            # negative learning
            loss_ += loss_ratio * unsup_loss_neg[0]
            loss_ += loss_ratio * unsup_loss_neg[1]



        # Back Propagation
        optimizer.zero_grad()
        # with torch.autograd.detect_anomaly():
        loss_.backward()
        optimizer.step()


        train_bar.desc = f'[Train Epoch {epoch}/{args.epochs}] ls: {loss_:.3f}, unsuploss: {unsup_loss[0]:.3f},' \
                         f'{unsup_loss[1]:.3f}, unsup_neg_loss: {unsup_loss_neg[0]:.3f}'




def Test2(model, test_loader, args, epoch, writer, best_acc=0, save_checkpoint=False):
    with torch.no_grad():
        print('==> Evaluation...')
        model.eval()
        top1_acc = [AverageMeter2("Top1") for i in range(7)]
        top5_acc = [AverageMeter2("Top5") for i in range(7)]

        # evaluate on webvision val set
        test_target_bar = tqdm(test_loader)
        for batch_index, (index, input, landmark, target, img_aug) in enumerate(test_target_bar):
            features, preds, label = model(input, landmark, target, img_aug, args, class_acc=None, is_eval=True)

            for i in range(7):
                acc1, acc5 = accuracy(preds[i], label, topk=(1, 5))
                top1_acc[i].update(acc1[0])
                top5_acc[i].update(acc5[0])

                # average across all processes


        for i in range(args.class_num):
            acc_tensors = torch.Tensor([top1_acc[i].avg, top5_acc[i].avg]).cuda(args.gpu)
            print('Accuracy is %.2f%% (%.2f%%)' % (acc_tensors[0], acc_tensors[1]))
            if i == 0:
                writer.add_scalars('Test/Acc_Target', {'global': acc_tensors[0]}, epoch)
            if i == 6:
                writer.add_scalars('Test/Acc_Target', {'global_local': acc_tensors[0]}, epoch)

        if save_checkpoint:
            global_acc = top1_acc[0].avg
            global_local_acc = top1_acc[6].avg

            if best_acc < global_acc or best_acc < global_local_acc:
                best_acc = max(global_acc, global_local_acc)
                print("***************")
                if global_acc > global_local_acc:
                    print(f'[Save] Best Acc: {best_acc:.4f}, the classifier is global_classifier. Save the checkpoint!')
                else:
                    print(f'[Save] Best Acc: {best_acc:.4f}, the classifier is global_local_classifier. Save the checkpoint!')
                print("***************")
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))
                else:
                    torch.save(model.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))

            return best_acc

    return

def Test3(model, test_loader, args, epoch, writer, best_acc=0, save_checkpoint=False):
    with torch.no_grad():
        print('==> Evaluation...')
        model.eval()
        top1_acc = [[AverageMeter() for i in range(7)] for i in range(7)]
        prec = [[AverageMeter() for i in range(7)] for i in range(7)]
        recall = [[AverageMeter() for i in range(7)] for i in range(7)]

        test_target_bar = tqdm(test_loader)
        for batch_index, (index, input, landmark, target, img_aug) in enumerate(test_target_bar):
            features, preds, label = model(input, landmark, target, img_aug, args, class_acc=None, is_eval=True)

            for classifier_id in range(7):
                Compute_Accuracy(args, preds[classifier_id], label, top1_acc[classifier_id], prec[classifier_id],
                                 recall[classifier_id])



        for classifier_id in range(7):
            Accuracy_Info, accs, prec_avg, recall_avg, f1_avg = Show_Accuracy(top1_acc[classifier_id],
                                                    prec[classifier_id], recall[classifier_id])
            print('Accuracy is %.2f%%' % (accs * 100.0))
            print('F1 is %.2f%%' % (f1_avg * 100.0))


        if save_checkpoint:
            for i in range(7):
                if i == 0:
                    writer.add_scalars('Test/Acc_Target', {'global': accs[i]}, epoch)
                if i == 6:
                    writer.add_scalars('Test/Acc_Target', {'global_local': accs[i]}, epoch)
            global_acc = accs[0] * 100.0
            global_local_acc = accs[6] * 100.0

            if best_acc < global_acc or best_acc < global_local_acc:
                best_acc = max(global_acc, global_local_acc)
                print("***************")
                if global_acc > global_local_acc:
                    print(f'[Save] Best Acc: {best_acc:.4f}, the classifier is global_classifier. Save the checkpoint!')
                else:
                    print(f'[Save] Best Acc: {best_acc:.4f}, the classifier is global_local_classifier. Save the checkpoint!')
                print("***************")
                if isinstance(model, nn.DataParallel):
                    torch.save(model.module.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))
                else:
                    torch.save(model.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))


            return best_acc


def init_p_model(model, train_loader,args):
    with torch.no_grad():
        print('==> Calculate...')
        model.eval()

        train_loader_bar = tqdm(train_loader)
        for batch_index, (index, input, landmark, target, img_aug) in enumerate(train_loader_bar):
            features, preds, label = model(input, landmark, target, img_aug, args, class_acc=None, is_init_pmodel=True)

        print('==> Calculate over...')
        model.print_p_model()



def main():
    """Main."""

    # Parse Argument
    # args = parser.parse_args()   # 构造参数
    args = construct_args()
    torch.manual_seed(args.seed)  # 人工种子
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

    if args.isTest:
        print('Test Model.')
    else:
        print('Learning Rate: %f' % args.lr)
        print('Momentum: %f' % args.momentum)
        print('Train Epoch: %d' % args.epochs)
        print('Weight Decay: %f' % args.weight_decay)

        if args.useAFN:
            print('Use AFN Loss: %s' % args.methodOfAFN)
            if args.methodOfAFN == 'HAFN':
                print('Radius of HAFN Loss: %f' % args.radius)
            else:
                print('Delta Radius of SAFN Loss: %f' % args.deltaRadius)
            print('Weight L2 nrom of AFN Loss: %f' % args.weight_L2norm)

        if args.useDAN:
            print('Use DAN Loss: %s' % args.methodOfDAN)
            print('Learning Rate(Adversarial Network): %f' % args.lr_ad)

    print('================================================')

    print('Number of classes : %d' % args.class_num)
    if not args.useLocalFeature:
        print('Only use global feature.')
    else:
        print('Use global feature and local feature.')

        if args.useIntraGCN:
            print('Use Intra GCN.')
        if args.useInterGCN:
            print('Use Inter GCN.')

        if args.useRandomMatrix and args.useAllOneMatrix:
            print('Wrong : Use RandomMatrix and AllOneMatrix both!')
            return None
        elif args.useRandomMatrix:
            print('Use Random Matrix in GCN.')
        elif args.useAllOneMatrix:
            print('Use All One Matrix in GCN.')

        if args.useCov and args.useCluster:
            print('Wrong : Use Cov and Cluster both!')
            return None
        else:
            if args.useCov:
                print('Use Mean and Cov.')
            else:
                print('Use Mean.') if not args.useCluster else print('Use Mean in Cluster.')

    print('================================================')

    print('================================================')
    # Bulid Model
    print('Building Model...')
    model = MoPro(args)
    # with open(args.OutputPath + "/second.log", "a") as f:
    #     num = 0
    #     for k, v in model.named_parameters():
    #         num += 1
    #         f.writelines(str(num) + "、" + str(k) + "\n" + str(v) + "\n\n")

    if args.Resume_Model != 'None':
        print('Resume Model: {}'.format(args.Resume_Model))
        checkpoint = torch.load(args.Resume_Model, map_location='cuda:0')
        model.load_state_dict(checkpoint, strict=False)
    else:
        print('No Resume Model')



    print('Done!')
    print('================================================')

    # Bulid Dataloder
    print("Building Train and Test Dataloader...")
    train_source_dataloader, _ = BulidDataloader(args, flag1='train', flag2='source')
    train_target_dataloader, init_train_dataset_data = BulidDataloader(args, flag1='train', flag2='target')
    test_source_dataloader, _ = BulidDataloader(args, flag1='test', flag2='source')
    test_target_dataloader, _ = BulidDataloader(args, flag1='test', flag2='target')
    labeled_train_target_loader = None
    print('Done!')
    print('================================================')


    # # Bulid Adversarial Network
    # print('Building Adversarial Network...')
    # # @ m21-11-13, 新写的对抗网络数组
    # random_layers, ad_nets = [], []
    # for i in range(2):
    #     random_layer, ad_net = BulidAdversarialNetwork(args, 64, args.class_num) if args.useDAN else (None, None)
    #     ad_nets.append(ad_net)
    #     random_layers.append(random_layer)
    # random_layer, ad_net = BulidAdversarialNetwork(args, 384, args.class_num) if args.useDAN else (None, None)
    # ad_nets.append(ad_net)
    # random_layers.append(random_layer)
    # print('Done!')
    # # ! m21-11-13, 注释
    # # random_layer, ad_net = BulidAdversarialNetwork(args, model.output_num(), args.class_num) if args.useDAN else (None, None)
    # print('================================================')


    # Set Optimizer #@ m21-11-13, 新增7个判别器的优化
    print('Building Optimizer...')
    # param_optim = Set_Param_Optim(args, model)
    # optimizer = Set_Optimizer(args, param_optim, args.lr, args.weight_decay, args.momentum)
    model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    print('Done!')
    print('================================================')

    # Init Mean #! m21-11-13, 注释
    '''
        if args.useLocalFeature and not args.isTest:

            if args.useCov:
                print('Init Mean and Cov...')
                Initialize_Mean_Cov(args, model, False)
            else:
                if args.useCluster:
                    print('Initialize Mean in Cluster....')
                    Initialize_Mean_Cluster(args, model, False, train_source_dataloader, train_target_dataloader)
                else:
                    print('Init Mean...')
                    Initialize_Mean(args, model, False)

            torch.cuda.empty_cache()

            print('Done!')
            print('================================================')
        '''

    # Save Best Checkpoint
    Best_Accuracy = 0


    # Running Experiment
    print("Run Experiment...")
    writer = SummaryWriter(os.path.join(args.OutputPath, 'visual_board'))

    if not args.isTest:
        labeled_train_target_loader_, confidence_, proportion_, collect_num, category_confidence_, category_proportion_ = BuildLabeledDataloader(args,
                                                                                    train_target_dataloader,
                                                                                    init_train_dataset_data,
                                                                                    model)

        labeled_train_target_loader = labeled_train_target_loader_
        confidence, proportion = confidence_, proportion_
        category_confidence, category_proportion = category_confidence_, category_proportion_
    for epoch in range(1, args.epochs + 1):

        if args.showFeature and epoch > 1:
            print(f"=================\ndraw the tSNE graph...")
            Visualization(args.OutputPath + '/result_pics/train/source/{}_Source.jpg'.format(epoch), model,
                          dataloader=train_source_dataloader, useClassify=True, domain='Source')
            Visualization(args.OutputPath + '/result_pics/train/target/{}_Target.jpg'.format(epoch), model,
                          train_target_dataloader, useClassify=True, domain='Target')
            #
            # VisualizationForTwoDomain(args.OutputPath + '/result_pics/train_tow_domain/{}_train'.format(epoch), model,
            #                           train_source_dataloader, train_target_dataloader, useClassify=False,
            #                           showClusterCenter=False)
            # VisualizationForTwoDomain(args.OutputPath + '/result_pics/test_tow_domain/{}_test'.format(epoch), model,
            #                           test_source_dataloader, test_target_dataloader, useClassify=False,
            #                           showClusterCenter=False)
            print(f"finish drawing!\n=================")

        if not args.isTest:
            if args.useCluster and epoch % 10 == 1:
                print(f"=================\nupdate the running_mean...")
                Initialize_Mean_Cluster(args, model, False, train_source_dataloader, train_target_dataloader)
                torch.cuda.empty_cache()
                print(f"finish the updating!\n=================")


            writer.add_scalars('Labeled_Prob_Confi', {'confidence': confidence, 'probility': proportion}, epoch)

            Train(args, model, train_source_dataloader, train_target_dataloader, labeled_train_target_loader, optimizer,
                  epoch, writer, collect_num)



        Test3(model, test_source_dataloader, args, epoch, writer)
        if args.isTest:
            Test3(model, test_target_dataloader, args, epoch, writer, Best_Accuracy, save_checkpoint=False)
            break
        else:
            Best_Accuracy = Test3(model, test_target_dataloader, args, epoch, writer, Best_Accuracy, save_checkpoint=True)


    writer.close()
    print(f"==========================\n{args.Log_Name} is done, ")
    print(f"saved in：{args.OutputPath}")


if __name__ == '__main__':
    main()