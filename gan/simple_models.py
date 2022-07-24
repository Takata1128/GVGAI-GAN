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


class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shape, use_self_attention):
        super().__init__()
        self.is_self_attention = use_self_attention
        self.resize = Resize(shape)
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.act = nn.LeakyReLU(True)
        self.bn = nn.BatchNorm2d(out_ch)
        if use_self_attention:
            self.attn = Self_Attn(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, label=None):
        x = self.resize(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.attn(x)
        x = self.act(x)
        return x


class DisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_self_attention, use_bn):
        super().__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.act = nn.LeakyReLU(True)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_ch)
        if use_self_attention:
            self.attn = Self_Attn(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, label=None):
        x = self.conv(x)
        x = self.act(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.attn(x)
        x = self.act(x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        out_dim: int,
        shapes: tuple[tuple[int]],
        z_shape,
        filters=128,
        use_self_attention=True,
        is_conditional=False,
    ):
        super(Generator, self).__init__()
        self.z_size = z_shape[0]
        self.use_self_attention = use_self_attention
        self.preprocess = nn.Sequential(
            nn.ConvTranspose2d(self.z_size, filters,
                               kernel_size=shapes[0], stride=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(True)
        )
        self.block1 = GenBlock(
            filters, filters//2, shapes[1], use_self_attention=False
        )
        self.block2 = GenBlock(
            filters//2, filters//4, shapes[2], use_self_attention=True
        )
        self.outconv = nn.Sequential(
            Resize(shapes[-1]),
            nn.Conv2d(filters//4, out_dim, 3, padding=1, bias=True),
            # Self_Attn(out_dim)  # additional
        )

    def forward(self, z, label=None):
        x = z.view(-1, z.shape[1], 1, 1)
        x = self.preprocess(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.outconv(x)
        return x

    def summary(self, batch_size=64):
        summary(self, (batch_size, self.z_size))


class Discriminator(nn.Module):
    def __init__(
        self,
        in_ch,
        shapes,
        filters=16,
        is_self_attention=True,
        is_minibatch_std=False,
        is_spectral_norm=False,
        is_conditional=False,
        use_recon_loss=False
    ):
        super(Discriminator, self).__init__()
        self.is_self_attention = is_self_attention
        self.use_minibatch_std = is_minibatch_std
        self.is_spectral_norm = is_spectral_norm
        self.use_recon_loss = use_recon_loss
        self.input_ch = in_ch
        self.input_shape = shapes[0]

        self.preprocess = nn.Sequential(
            Self_Attn(in_ch),  # additional
            nn.Conv2d(in_ch, filters, 3, stride=1,
                      padding=1), nn.LeakyReLU(True)
        )
        self.block1 = DisBlock(
            filters, filters*2, use_self_attention=False, use_bn=False)
        self.block2 = DisBlock(
            filters*2, filters*4, use_self_attention=True, use_bn=False
        )

        self.decoder = nn.Sequential(
            GenBlock(filters*4, filters*2,
                     shapes[1], use_self_attention=False),
            GenBlock(filters*2, filters, shapes[0], use_self_attention=True),
            nn.Conv2d(filters, self.input_ch, 3, padding=1, bias=True),
            nn.Softmax2d()
        )

        if self.use_minibatch_std:
            self.minibatch_std = MiniBatchStd()
        self.postprocess = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(filters*4 + int(is_minibatch_std), 1)

    def forward(self, x, label=None):
        x = self.preprocess(x)
        x = self.block1(x)
        x = self.block2(x)
        branch_x = x
        if self.use_minibatch_std:
            x = self.minibatch_std(x)
        x = self.postprocess(x)
        x = x.view(x.size(0), -1)
        out = self.output(x)
        recon = None
        if self.use_recon_loss:
            recon = self.decoder(branch_x)
            return out, recon
        return out

    def summary(self, batch_size=64):
        summary(self, (batch_size, self.input_ch, *self.input_shape))
