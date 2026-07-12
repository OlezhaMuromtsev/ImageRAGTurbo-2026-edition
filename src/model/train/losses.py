import torch
import torch.nn.functional as F
import lpips

def loss_adv_D(score_real, score_fake):
    return F.relu(1 - score_real).sum() + F.relu(1 + score_fake).sum()

def loss_adv_G(score):
    return -score.sum()

def loss_distill(fake, real):
    return F.smooth_l1_loss(fake, real)

def loss_elpips(fake, real):
    loss_fn = lpips.LPIPS(net='vgg')
    return loss_fn(fake, real) # возможно, понадобится нормализовать вектора
