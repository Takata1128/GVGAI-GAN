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


class ConditionalSelfAttention(nn.Module):
    def __init__(self, in_dim, shape, label_dim):
        super().__init__()
        self.shape = shape
        self.attn = Self_Attn(in_dim)
        self.embedding = nn.Linear(label_dim, mul(*shape))

    def forward(self, x, label=None):
        x = self.attn(x)
        u = self.embedding(label)
        u = u.view(-1, 1, *self.shape)
        x = torch.cat([x, u], dim=1)
        return x


class MiniBatchStd(nn.Module):
    def forward(self, x):
        std = torch.std(x, dim=0, keepdim=True)
        mean = torch.mean(std, dim=(1, 2, 3), keepdim=True)
        n, c, h, w = x.shape
        mean = torch.ones(n, 1, h, w, dtype=x.dtype, device=x.device) * mean
        return torch.cat((x, mean), dim=1)


class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn, use_deconv=False):
        super().__init__()
        self.use_bn = use_bn
        self.use_deconv = use_deconv
        if self.use_deconv:
            self.upsample = nn.ConvTranspose2d(
                in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        else:
            self.upsample = nn.Sequential(nn.Upsample(
                scale_factor=2), nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        if use_bn:
            self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(True)

    def forward(self, x, label=None):
        x = self.upsample(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class DisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn):
        super().__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        if use_bn:
            self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(True)

    def forward(self, x, label=None):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        out_dim: int,
        shapes: list[tuple[int]],
        z_shape,
        filters=32,
        use_conditional=False
    ):
        super(Generator, self).__init__()
        self.z_size = z_shape[0]
        self.init_filters = filters
        self.init_shape = shapes[0]
        self.use_conditional = use_conditional
        self.out_dim = out_dim

        self.preprocess = nn.Linear(
            self.z_size, mul(filters, mul(*self.init_shape)))

        self.block1 = GenBlock(
            filters, filters//2, True
        )
        self.self_attn0 = Self_Attn(filters//2)
        self.block2 = GenBlock(
            filters//2, filters//4, True
        )
        if self.use_conditional:
            self.self_attn = ConditionalSelfAttention(
                filters//4, shapes[-1], out_dim)
        else:
            self.self_attn = Self_Attn(filters//4)

        self.outconv = nn.Conv2d(filters//4+int(self.use_conditional), out_dim,
                                 kernel_size=1, stride=1)

    def forward(self, z, label=None):
        x = self.preprocess(z)
        x = x.view(-1, self.init_filters, *self.init_shape)
        x = self.block1(x)
        x = self.self_attn0(x)
        x = self.block2(x)
        if self.use_conditional:
            x = self.self_attn(x, label)
        else:
            x = self.self_attn(x)
        x = self.outconv(x)
        return x

    def summary(self, batch_size=64):
        if self.use_conditional:
            summary(self, ((batch_size, self.z_size),
                    (batch_size, self.out_dim)))
        else:
            summary(self, (batch_size, self.z_size),)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_ch,
        shapes,
        filters=16,
        is_minibatch_std=False,
        use_recon_loss=False,
        use_conditional=False
    ):
        super(Discriminator, self).__init__()
        self.use_minibatch_std = is_minibatch_std
        self.use_recon_loss = use_recon_loss
        self.use_conditional = use_conditional
        self.input_ch = in_ch
        self.input_shape = shapes[0]
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_ch, filters, 1, stride=1, bias=True),
            nn.LeakyReLU(True),
        )
        if self.use_conditional:
            self.self_attn = ConditionalSelfAttention(
                filters, self.input_shape, self.input_ch)
        else:
            self.self_attn = Self_Attn(filters)
        self.decoder = Decoder(filters, self.input_ch)

        self.block1 = DisBlock(
            filters+int(self.use_conditional), filters*2, use_bn=False)
        self.self_attn0 = Self_Attn(filters*2)
        self.block2 = DisBlock(filters*2, filters*4, use_bn=False)

        if self.use_minibatch_std:
            self.minibatch_std = MiniBatchStd()
        self.postprocess = nn.AdaptiveAvgPool2d(1)
        self.output = nn.Linear(filters*4 + int(is_minibatch_std), 1)

    def forward(self, x, label=None):
        x = self.preprocess(x)
        if self.use_conditional:
            x = self.self_attn(x, label)
        else:
            x = self.self_attn(x)
        x = self.block1(x)
        x = self.self_attn0(x)
        x = self.block2(x)
        branch_x = x
        if self.use_minibatch_std:
            x = self.minibatch_std(x)
        x = self.postprocess(x)
        x = x.view(x.size(0), -1)
        out = self.output(x)
        recon = None
        if self.use_recon_loss:
            if self.use_conditional:
                recon = self.decoder(branch_x, label)
            else:
                recon = self.decoder(branch_x)
            return out, recon
        return out

    def summary(self, batch_size=64):
        if self.use_conditional:
            summary(self, ((batch_size, self.input_ch, *self.input_shape),
                    (batch_size, self.input_ch)))
        else:
            summary(self, (batch_size, self.input_ch, *self.input_shape))


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.block1 = GenBlock(in_channel*4, in_channel*2, use_bn=True)
        self.block2 = GenBlock(in_channel*2, in_channel, use_bn=True)
        self.attn = Self_Attn(in_channel)
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=1, stride=1, bias=True)
        self.softmax = nn.Softmax2d()

    def forward(self, x, label=None):
        x = self.block1(x)
        x = self.block2(x)
        x = self.attn(x)
        x = self.conv(x)
        return self.softmax(x)
