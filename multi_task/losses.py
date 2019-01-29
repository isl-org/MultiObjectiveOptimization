import torch
import torch.nn.functional as F 

def nll(pred, gt, val=False):
    if val:
        return F.nll_loss(pred, gt, size_average=False)
    else:
        return F.nll_loss(pred, gt)

def rmse(pred, gt, val=False):
    pass

def cross_entropy2d(input, target, weight=None, val=False):
    if val:
        size_average = False
    else:
        size_average = True 
    
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

def l1_loss_depth(input, target, val=False):
    if val:
        size_average = False
    else:
        size_average = True
    mask = target > 0
    if mask.data.sum() < 1:
        # no instance pixel
        return None 

    lss = F.l1_loss(input[mask], target[mask], size_average=False)
    if size_average:
        lss = lss / mask.data.sum()
    return lss 


def l1_loss_instance(input, target, val=False):
    if val:
        size_average = False
    else:
        size_average = True
    mask = target!=250
    if mask.data.sum() < 1:
        # no instance pixel
        return None 

    lss = F.l1_loss(input[mask], target[mask], size_average=False)
    if size_average:
        lss = lss / mask.data.sum()
    return lss 

def get_loss(params):
    if 'mnist' in params['dataset']:
        loss_fn = {}
        for t in params['tasks']:
            loss_fn[t] = nll 
        return loss_fn

    if 'cityscapes' in params['dataset']:
        loss_fn = {}
        if 'D' in params['tasks']:
            loss_fn['D'] = rmse
        if 'S' in params['tasks']:
            loss_fn['S'] = cross_entropy2d
        if 'I' in params['tasks']:
            loss_fn['I'] = l1_loss_instance
        if 'D' in params['tasks']:
            loss_fn['D'] = l1_loss_depth
        return loss_fn

    if 'celeba' in params['dataset']:
        loss_fn = {}
        for t in params['tasks']:
            loss_fn[t] = nll
        return loss_fn