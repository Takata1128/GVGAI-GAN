from __future__ import annotations
import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from torchinfo import summary


class Resize(nn.Module):
    def __init__(self, shape):
        super(Resize, self).__init__()
        self.shape = shape

    def forward(self, x):
        return nn.functional.interpolate(
            x, size=self.shape, mode="nearest"
        )  # mode='bilinear', align_corners=True)


class Self_Attn(nn.Module):
    """
    Self Attention Layer
    """

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.key_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1
        )
        self.value_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1
        )
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        inputs :
            x : input feature map (B,C,W,H)
        returns :
            out : self attention value + input feature
            attention : (B,WH,WH)
        """

        b, c, w, h = x.shape
        proj_query = self.query_conv(x).view(
            b, -1, w * h).permute(0, 2, 1)  # (B,WH,C)
        proj_key = self.key_conv(x).view(b, -1, w * h)
        energy = torch.bmm(proj_query, proj_key)  # mat mul by batch
        attention = self.softmax(energy)  # (B,WH,WH)
        proj_value = self.value_conv(x).view(b, -1, w * h)  # (B,C,WH)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, -1, w, h)

        out = self.gamma * out + x
        return out


class MiniBatchStd(nn.Module):
    def forward(self, x):
        std = torch.std(x, dim=0, keepdim=True)
        mean = torch.mean(std, dim=(1, 2, 3), keepdim=True)
        n, c, h, w = x.shape
        mean = torch.ones(n, 1, h, w, dtype=x.dtype, device=x.device) * mean
        return torch.cat((x, mean), dim=1)


class Generator(nn.Module):
    def __init__(
        self,
        out_dim: int,
        shapes: list[tuple[int]],
        z_shape,
        filters=8,
        use_self_attention=True
    ):
        super(Generator, self).__init__()
        self.z_size = z_shape[0]
        self.init_filters = 4
        self.output_shape = shapes[-1]
        self.use_self_attention = use_self_attention

        self.preprocess = nn.Linear(
            self.z_size, mul(self.init_filters, mul(*self.output_shape)))
        self.conv0 = nn.ConvTranspose2d(self.init_filters, filters, 1, 1, 0)
        self.bn0 = nn.BatchNorm2d(filters)
        self.conv1 = nn.ConvTranspose2d(filters, filters*2, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(filters*2)
        self.attn = Self_Attn(filters*2)
        self.conv2 = nn.ConvTranspose2d(filters*2, filters, 1, 1, 0)
        self.bn2 = nn.BatchNorm2d(filters)
        self.output = nn.ConvTranspose2d(filters, out_dim, 1, 1, 0)

        self.act = nn.LeakyReLU()

    def forward(self, z, label=None):
        x = self.preprocess(z)
        x = x.view(-1, self.init_filters, *self.output_shape)
        x = self.act(self.bn0(self.conv0(x)))
        x = self.act(self.bn1(self.conv1(x)))
        if self.use_self_attention:
            x = self.attn(x)
        x = self.act(self.bn2(self.conv2(x)))
        x = self.output(x)
        return x

    def summary(self, batch_size=64):
        summary(self, (batch_size, self.z_size))


class Discriminator(nn.Module):
    def __init__(
        self,
        in_ch,
        shapes,
        filters=16,
        use_self_attention=True,
        use_recon_loss=False
    ):
        super(Discriminator, self).__init__()
        self.use_self_attention = use_self_attention
        self.use_recon_loss = use_recon_loss
        self.input_ch = in_ch
        self.input_shape = shapes[0]

        self.conv1 = nn.Conv2d(in_ch, filters, 1, 1, 0)
        self.conv2 = nn.Conv2d(filters, filters*2, 1, 1, 0)
        self.attn1 = Self_Attn(filters*2)
        self.conv3 = nn.Conv2d(filters*2, filters*2, 4, 2, 1)
        self.attn2 = Self_Attn(filters*2)
        self.conv4 = nn.Conv2d(filters*2, filters, 4, 2, 1)

        self.act = nn.LeakyReLU()
        self.reconstruct = nn.Sequential(
            nn.ConvTranspose2d(filters, filters*2, 4, 2, 1),
            nn.BatchNorm2d(filters*2),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(filters*2, filters*2, 4, 2, 1),
            nn.BatchNorm2d(filters*2),
            nn.LeakyReLU(),
            nn.Conv2d(filters*2, filters, 1, 1, 0),
            nn.BatchNorm2d(filters),
            nn.LeakyReLU(),
            nn.Conv2d(filters, in_ch, 1, 1, 0),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(filters, 1)

    def forward(self, x, label=None):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        if self.use_self_attention:
            x = self.attn1(x)
        x = self.act(self.conv3(x))
        if self.use_self_attention:
            x = self.attn2(x)
        x = self.act(self.conv4(x))
        x_r = x
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        out = self.output(x)
        if self.use_recon_loss:
            recon = self.reconstruct(x_r)
            return out, recon
        return out

    def summary(self, batch_size=64):
        summary(self, (batch_size, self.input_ch, *self.input_shape))
