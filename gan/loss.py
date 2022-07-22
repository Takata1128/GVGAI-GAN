import torch
import torch.nn as nn


def d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, flip_prob: float = 0.0):
    # loss from real images
    p_real = torch.sigmoid(real_logits)
    real_labels = torch.full_like(real_logits, fill_value=1.0-flip_prob)
    real_labels = torch.bernoulli(real_labels).to(device=real_logits.device)
    # real_labels = torch.ones(len(real_logits)).to(device=real_logits.device)
    loss_real = torch.nn.BCELoss()(p_real, real_labels)
    D_x = p_real.mean().item()

    # loss from fake images
    p_fake = torch.sigmoid(fake_logits)
    zeros = torch.zeros(len(fake_logits)).to(device=fake_logits.device)
    loss_fake = torch.nn.BCELoss()(p_fake, zeros)
    D_G_z = p_fake.mean().item()

    return loss_real, loss_fake, D_x, D_G_z


def g_loss(fake_logits: torch.Tensor):
    ones = torch.ones(len(fake_logits)).to(device=fake_logits.device)
    p_fake = torch.sigmoid(fake_logits)

    generator_loss = torch.nn.BCELoss()(p_fake, ones)

    return generator_loss


def div_loss(fake: torch.Tensor, loss_type: str, lambda_div=1.0):
    if loss_type == "l1":
        return -torch.abs(fake[1:] - fake[:-1]).mean() * lambda_div
    elif loss_type == "l2":
        return -((fake[1:] - fake[:-1]) ** 2).mean() * lambda_div
    else:
        return torch.tensor(0)


def d_loss_hinge(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    loss_real = torch.relu(1.0 - real_logits).mean()
    loss_fake = torch.relu(1.0 + fake_logits).mean()
    D_x = torch.sigmoid(real_logits).mean().item()
    D_G_z = torch.sigmoid(fake_logits).mean().item()
    return loss_real, loss_fake, D_x, D_G_z


def g_loss_hinge(fake_logits: torch.Tensor):
    discriminator_loss = -torch.mean(fake_logits)
    return discriminator_loss
