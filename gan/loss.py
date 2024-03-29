from __future__ import annotations
import torch
import torch.nn as nn
from gan.game.env import Game


def d_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor, smooth_label_value: float = 0.1):
    # loss from real images
    p_real = torch.sigmoid(real_logits)
    real_labels = torch.full_like(real_logits, fill_value=smooth_label_value)
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


def div_loss(latent: torch.Tensor, fake: torch.Tensor, hiddens: list[torch.Tensor], loss_type: str, lambda_div=1.0, game: Game = None):
    if loss_type is None:
        return torch.tensor(0)
    ls = latent.size(0)
    first = fake[:ls // 2, :, :game.height, :game.width]
    second = fake[ls // 2:, :, :game.height, :game.width]
    if loss_type == "l1":
        # return -torch.abs(first - second).mean() * lambda_div
        lz = torch.abs(first - second).mean()
        eps = 1 * 1e-5
        return (1 / (lz + eps)) * lambda_div
    elif loss_type == 'l1-hidden':
        return -torch.abs(hiddens[1][1:] - hiddens[1][:-1]).mean() * lambda_div
    elif loss_type == 'l1-hidden-latent':
        return -torch.abs(hiddens[1][1:] - hiddens[1][:-1]).mean() / torch.abs(latent[1:] - latent[:-1]).mean() * lambda_div
    elif loss_type == 'l1-latent':
        # return -(torch.abs(first - second).mean() / torch.abs(latent[:ls // 2] - latent[ls // 2:]).mean()) * lambda_div
        lz = torch.abs(first - second).mean() / \
            torch.abs(latent[:ls // 2] - latent[ls // 2:]).mean()
        eps = 1 * 1e-5
        return (1 / (lz + eps)) * lambda_div
    elif loss_type == "l2":
        return -((first - second) ** 2).mean() * lambda_div
    else:
        raise NotImplementedError()


def recon_loss(recon: torch.Tensor, real: torch.Tensor):
    recon = torch.nn.Softmax2d()(recon)
    return torch.nn.L1Loss()(recon, real)


def d_loss_hinge(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    loss_real = torch.relu(1.0 - real_logits).mean()
    loss_fake = torch.relu(1.0 + fake_logits).mean()
    D_x = real_logits.mean().item()
    D_G_z = fake_logits.mean().item()
    return loss_real, loss_fake, D_x, D_G_z


def g_loss_hinge(fake_logits: torch.Tensor):
    generator_loss = -torch.mean(fake_logits)
    return generator_loss


def d_loss_wgan(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    loss_real = -torch.mean(real_logits)
    loss_fake = torch.mean(fake_logits)
    D_x = real_logits.mean().item()
    D_G_z = fake_logits.mean().item()
    return loss_real, loss_fake, D_x, D_G_z


def g_loss_wgan(fake_logits: torch.Tensor):
    generator_loss = -torch.mean(fake_logits)
    return generator_loss


def d_loss_lsgan(real_logits: torch.Tensor, fake_logits: torch.Tensor):
    loss_real = 0.5 * nn.MSELoss()(real_logits, torch.ones_like(real_logits) * (1.0))
    loss_fake = 0.5 * nn.MSELoss()(fake_logits, torch.ones_like(fake_logits) * (-1.0))
    D_x = real_logits.mean().item()
    D_G_z = fake_logits.mean().item()
    return loss_real, loss_fake, D_x, D_G_z


def g_loss_lsgan(fake_logits: torch.Tensor):
    generator_loss = nn.MSELoss()(fake_logits, torch.ones_like(fake_logits) * 0.0)
    return generator_loss


def calc_gradient_penalty(real_images: torch.Tensor, fake_images: torch.Tensor, discriminator: nn.Module):
    batch_size = real_images.size()[0]
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(real_images).to(real_images.device)
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    prob_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=prob_interpolated, inputs=interpolated, grad_outputs=torch.ones(prob_interpolated.size()).to(real_images.device), create_graph=True, retain_graph=True)[0]
    gradients_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradients_norm - 1)**2).mean()
    return gradient_penalty
