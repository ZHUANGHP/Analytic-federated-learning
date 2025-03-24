import argparse
import os
import random
import shutil
import time
import warnings
from enum import Enum

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.models
from torch.utils.data import Subset
from dataset import prepare_data
import numpy as np
from afl import LinearAnalytic, init_local, local_update, aggregation, clean_regularization, validate

# Basic Setup
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', metavar='DIR', nargs='?', default='cifar100',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
#Data Setup
parser.add_argument('--data', default='./data', type=str, metavar='PATH',
                    help='path of dataset')
parser.add_argument('--datadir', default='./dataset', type=str, metavar='PATH',
                    help='path to locate dataset split')
parser.add_argument( '--seed', default=1, type=int, metavar='N',
                    help='random seed for spliting data')
parser.add_argument( '--modelseed', default=1, type=int, metavar='N',
                    help='random seed for spliting data')
parser.add_argument( '--num_clients', default=50, type=int, metavar='N',
                    help='number of clients')
parser.add_argument( '--num_classes', default=100, type=int, metavar='N',
                    help='total number of classes')
parser.add_argument( '--niid',  dest='niid', action='store_true',
                    help='set data to non-iid')
parser.add_argument( '--balance',  dest='balance', action='store_true',
                    help='balance distribution')
parser.add_argument('--partition', default='dir', type=str,
                    help='dirstribution type of non-iid setting')
parser.add_argument( '--alpha', default=0.1, type=float,
                    help='skewness of dir distribution')
parser.add_argument( '--shred', default=10, type=int,
                    help='skewness of pat distribution')
parser.add_argument( '--rg', default=0, type=float,
                    help='regularization factor of analytic learning')
parser.add_argument( '--clean_reg',  dest='clean_reg', action='store_true',
                    help='clean regularization factor after aggregation')
best_acc1 = 0

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.modelseed)
        torch.manual_seed(args.modelseed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        # Simply call main_worker function
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    import resnet
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model_pretrain = torchvision.models.__dict__[args.arch](pretrained=True)
        model = resnet.__dict__[args.arch](args.num_classes)
        model.load_state_dict(model_pretrain.state_dict(), strict=False)
        args.feat_size = model_pretrain.fc.weight.size(1)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = resnet.__dict__[args.arch]()
    model = model.cuda(args.gpu)

    # dataset spliting
    train_total, train_data_idx, testset = prepare_data(args)
    random.seed(args.modelseed)
    torch.manual_seed(args.modelseed)
    train_dataset = []
    for idx in range(args.num_clients):
        train_dataset_idx = torch.utils.data.Subset(train_total, train_data_idx[idx])
        train_dataset.append(train_dataset_idx)
    model.eval()

    global_model = LinearAnalytic(args.feat_size, args.num_classes).cuda(args.gpu)
    local_weights, local_R, local_C = [], [], []
    local_models = []
    local_train_acc = []
    print("Training locally!")
    start = time.time()
    for idx in range(args.num_clients):
        train_loader = torch.utils.data.DataLoader(train_dataset[idx],args.batch_size,drop_last=False,shuffle=True,num_workers=8)
        #train locally
        W, R, C = local_update(train_loader,model,global_model,args)
        local_model = init_local(args)
        local_model.fc.weight = torch.nn.parameter.Parameter(torch.t(W.float()))
        local_models.append(local_model)
        correct, num_sample = validate(train_loader, model, local_model.cuda(), args)
        acc = correct / num_sample
        W = W.cpu()
        # correct_val, num_sample_val = validate(val_loader, model, local_model, args)
        # acc_val = correct_val / num_sample_val
        print("Training Accuracy at training set on Client #{}: {}%".format(idx, acc * 100))
        local_weights.append(W)
        local_R.append(R)
        local_C.append(C)
        local_train_acc.append(acc.cpu().item())
    endtime = time.time() - start
    print("Elapsing time for local training: {}%".format(endtime))
    #aggregation
    print("Aggregating!")
    global_weight, global_R, global_C = aggregation(local_weights, local_R, local_C, args)
    print('Aggregation done!')
    global_model.fc.weight = torch.nn.parameter.Parameter(torch.t(global_weight.float()))

    #Evaluate the global model
    print("Evaluating global model!")


    val_loader = torch.utils.data.DataLoader(testset, args.batch_size, drop_last=False, shuffle=True, num_workers=8)
    correct, num_sample = validate(val_loader, model, global_model,args)
    acc = correct/num_sample * 100
    endtime_1 = time.time()-start
    print("Elapsing time for training and aggregation: {}%".format(endtime_1))
    print("Average accuracy on all Client: {}%".format(acc))
    acc_c = -100
    if args.clean_reg:
        global_weight_clean = clean_regularization(global_weight, global_C, args)
        global_model.fc.weight = torch.nn.parameter.Parameter(torch.t(global_weight_clean.float()))
        val_loader = torch.utils.data.DataLoader(testset, args.batch_size, drop_last=False, shuffle=True)
        correct_c, num_sample = validate(val_loader, model, global_model, args)
        acc_c = correct_c / num_sample * 100
        print("Average accuracy after regularization cleaning on all Client: {}%".format(acc_c))

    endtime_2 = time.time() - start
    print("Elapsing time plus cleansing regularization: {}%".format(endtime_2))
    import csv
    with open(args.dataset + args.arch + '_' + str(args.num_clients) + '_' + str(args.alpha) + '_' + str(args.shred) + '_' + str(args.partition)+ '.csv', mode='a+', encoding="ISO-8859-1",
              newline='') as file:
        data = (str(local_train_acc),) +  ('-',)+(str(acc.cpu().item()),)  + (str(acc_c.cpu().item()),) +  ('-',)   + (str(endtime),)  + (str(endtime_1),)  + (str(endtime_2),)+  ('-',) + (str(args),)
        wr = csv.writer(file)
        wr.writerow(data)
    print('written it to a csv file named {}.'.format(args.dataset + args.arch + '_' + str(args.num_clients) + '_' + str(args.alpha) + '.csv'))


if __name__ == '__main__':
    main()
