import torch.nn.functional as F

def kl_loss_compute(pred, soft_targets, reduce=True):
    kl = F.kl_div(
        F.log_softmax(pred, dim=1),
        F.softmax(soft_targets, dim=1),
        reduction="none",
    )

    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)
