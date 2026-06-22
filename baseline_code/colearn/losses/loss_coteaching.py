import torch
import torch.nn.functional as F
import numpy as np


def loss_coteaching(y_1, y_2, t, forget_rate):
    loss_1 = F.cross_entropy(y_1, t, reduction="none")
    ind_1_sorted = np.argsort(loss_1.detach().cpu().numpy())

    loss_2 = F.cross_entropy(y_2, t, reduction="none")
    ind_2_sorted = np.argsort(loss_2.detach().cpu().numpy())

    remember_rate = 1 - forget_rate
    num_remember = max(1, int(remember_rate * len(loss_1)))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    if torch.isnan(loss_1_update):
        print(loss_1_update, y_1[ind_2_update], t[ind_2_update], len(loss_1), remember_rate)

    return loss_1_update, loss_2_update
