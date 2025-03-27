import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class LinearAnalytic(nn.Module):
    def __init__(self, in_d, num_classes):
        super(LinearAnalytic, self).__init__()
        self.act = nn.Identity()
        self.fc = nn.Linear(in_d, num_classes, bias=False)

    def forward(self, x):
        x_act = self.act(x)
        x_fc = self.fc(x_act)
        return x_act, x_fc

def init_local(args):
    local_model = LinearAnalytic(args.feat_size, args.num_classes).cuda()
    return local_model

def local_update(train_loader,model,global_model,args):
    """
    Input: Trainloader that contains data of a client, backbone and args.
    Returns the weight of a client via analytic learning.
    """
    corr_rep = torch.zeros(args.feat_size, args.feat_size).cuda(args.gpu, non_blocking=True)
    corr_label = torch.zeros(args.feat_size, args.num_classes).cuda(args.gpu, non_blocking=True)
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(train_loader):

            images, labels = images.cuda(args.gpu, non_blocking=True), labels.cuda(args.gpu, non_blocking=True)
            # analytic learning
            reps = model(images)
            reps,_ = global_model(reps)
            label_onehot = F.one_hot(labels, args.num_classes).float()
            corr_rep += torch.t(reps) @ reps
            corr_label += torch.t(reps) @ (label_onehot)
        # matrix inverse with reqularization
        R = torch.inverse(torch.from_numpy(np.mat((corr_rep.cpu().numpy())+args.rg*np.eye(corr_rep.size(0)))).double()).cuda(args.gpu, non_blocking=True)
        Delta = R @ corr_label.double()
        # model.fc.weight = torch.nn.parameter.Parameter(torch.t(Delta.float()))
        # W = torch.t(model.fc.weight)
        W = Delta.double()
        R = R.double()
        C = torch.inverse(R).double().cpu()
        R = R.cpu()
    return W, R, C

def aggregation(W, R, C, args):
    """
    Input: List of the weights, R and C of all clients.
    Returns the average of the weights.
    """
    if len(W) < 2:
        print("No need to aggregate")
        return W[0].cuda(args.gpu, non_blocking=True), R[0].cuda(args.gpu, non_blocking=True), C[0].cuda(args.gpu, non_blocking=True)
    R[0] = R[0].cuda()
    C[0] = C[0].cuda()
    W[0] = W[0].cuda()
    R[1] = R[1].cuda()
    C[1] = C[1].cuda()
    W[1] = W[1].cuda()
    Wt = (torch.eye(R[0].shape[0]).double().cuda(args.gpu, non_blocking=True) - R[0] @ C[1] + R[0] @ C[
        1] @ torch.inverse(C[0] + C[1]) @ C[1]) @ W[0] + (
                 torch.eye(R[0].shape[0]).double().cuda(args.gpu, non_blocking=True) - R[1] @ C[0] + R[1] @ C[
             0] @ torch.inverse(C[0] + C[1]) @ C[0]) @ W[
             1]
    Ct = C[0] + C[1]
    Rt = torch.inverse(Ct)

    for i in range(1, len(W) - 1):
        R[i + 1] = R[i + 1].cuda()
        C[i + 1] = C[i + 1].cuda()
        W[i + 1] = W[i + 1].cuda()
        Wt = (torch.eye(R[0].shape[0]).double().cuda(args.gpu, non_blocking=True) - Rt @ C[i + 1] + Rt @ C[
            i + 1] @ torch.inverse(Ct + C[i + 1]) @ C[
                  i + 1]) @ Wt + (
                     torch.eye(R[0].shape[0]).double().cuda(args.gpu, non_blocking=True) - R[i + 1] @ Ct + R[
                 i + 1] @ Ct @ torch.inverse(
                 Ct + C[i + 1]) @ Ct) @ W[i + 1]
        Ct = Ct + C[i + 1]
        Rt = torch.inverse(Ct)
        R[i + 1] = R[i + 1].cpu()
        C[i + 1] = C[i + 1].cpu()
        W[i + 1] = W[i + 1].cpu()
    return Wt, Rt, Ct

def clean_regularization(W,C,args):
    """
        Identical implementation as the paper with the transformation by woodbury identity.
        """
    R_origin = torch.inverse(C - args.num_clients * args.rg * torch.eye(args.feat_size).float().cuda(args.gpu, non_blocking=True))
    Wt = W + (args.num_clients*args.rg*R_origin) @ W
    return Wt

def validate(val_loader, model, global_model,args):
    num_correct = 0
    num_sample = 0
    model.eval()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):

            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            ref = model(images)
            _,output = global_model(ref)
            # measure accuracy and record loss
            num_correct += correct(output, target)
            num_sample += images.size(0)
    return num_correct, num_sample

def correct(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # maxk = max(1)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return correct[:1].reshape(-1).float().sum(0, keepdim=False)