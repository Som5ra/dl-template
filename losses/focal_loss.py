import torch

def focal_loss(pred, gt):
    '''
    Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4).float()
    # clamp min value is set to 1e-12 to maintain the numerical stability
    pred = torch.clamp(pred, 1e-6).float()

    pos_loss = torch.log(torch.clamp(pred, min=1e-6)) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(torch.clamp(1 - pred, min=1e-6)) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.float().sum()
    neg_loss = neg_loss.float().sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss