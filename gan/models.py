import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import reduce
from operator import mul
from torchinfo import summary


def init_lecun_normal(tensor, scale=1.0):
    fan_in = torch.nn.init._calculate_correct_fan(tensor, "fan_in")
    std = scale * np.sqrt(1.0 / fan_in)
    with torch.no_grad():
        return tensor.normal_(0, std)


def lecun_init(layer, gain=1):
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        init_lecun_normal(layer.weight, gain)
        nn.init.zeros_(layer.bias)
    else:
        init_lecun_normal(layer.weight_ih_l0, gain)
        init_lecun_normal(layer.weight_hh_l0, gain)
        nn.init.zeros_(layer.bias_ih_l0)
        nn.init.zeros_(layer.bias_hh_l0)
    return layer


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


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
        proj_query = self.query_conv(x).view(b, -1, w * h).permute(0, 2, 1)  # (B,WH,C)
        proj_key = self.key_conv(x).view(b, -1, w * h)
        energy = torch.bmm(proj_query, proj_key)  # mat mul by batch
        attention = self.softmax(energy)  # (B,WH,WH)
        proj_value = self.value_conv(x).view(b, -1, w * h)  # (B,C,WH)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, -1, w, h)

        out = self.gamma * out + x
        return out


class Conditional_Self_Attn(nn.Module):
    """
    Conditional Self Attention Layer
    """

    def __init__(self, shape, label_channel):
        super(Conditional_Self_Attn, self).__init__()
        self.shape = shape
        self.label_channel = label_channel
        self.self_attn = Self_Attn(in_dim=shape[0])
        self.embeding = nn.Embedding(12 * 16, shape[1] * shape[2])

    def forward(self, x, y):
        x = self.self_attn(x)
        embedd = self.embeding(y).view(-1, self.label_channel, *self.shape[1:])
        x = torch.cat((x, embedd), dim=1)
        return x


class MiniBatchStd(nn.Module):
    def forward(self, x):
        std = torch.std(x, dim=0, keepdim=True)
        mean = torch.mean(std, dim=(1, 2, 3), keepdim=True)
        n, c, h, w = x.shape
        mean = torch.ones(n, 1, h, w, dtype=x.dtype, device=x.device) * mean
        return torch.cat((x, mean), dim=1)


class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shape, is_self_attention, is_conditional):
        super().__init__()
        self.is_self_attention = is_self_attention
        self.is_conditional = is_conditional

        self.resize = Resize(shape)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.act = nn.LeakyReLU(True)
        self.bn = nn.BatchNorm2d(out_ch)

        if is_conditional:
            self.attn = Conditional_Self_Attn((out_ch, *shape), 8)
        elif is_self_attention:
            self.attn = Self_Attn(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, label=None):
        x = self.resize(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        if self.is_conditional:
            x = self.attn(x, label)
        else:
            x = self.attn(x)
        return x


class DisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shape, is_self_attention, is_conditional):
        super().__init__()
        self.is_self_attention = is_self_attention
        self.is_conditional = is_conditional

        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        self.act = nn.LeakyReLU(True)
        if is_conditional:
            self.attn = Conditional_Self_Attn((out_ch, *shape), label_channel=8)
        elif is_self_attention:
            self.attn = Self_Attn(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, label=None):
        x = self.conv(x)
        x = self.act(x)
        if self.is_conditional:
            x = self.attn(x, label)
        else:
            x = self.attn(x)
        return x


class Generator(nn.Module):
    def __init__(
        self,
        out_dim,
        shapes,
        z_shape,
        filters=256,
        is_self_attention=True,
        is_conditional=False,
    ):
        super(Generator, self).__init__()
        self.z_size = z_shape[0]
        self.is_self_attention = is_self_attention

        self.init_shape = (filters, *shapes[0])
        self.preprocess = nn.Sequential(
            nn.Linear(self.z_size, reduce(mul, self.init_shape), bias=False),
            nn.LeakyReLU(True),
        )

        self.blocks = nn.ModuleList()
        in_ch = filters
        out_ch = filters
        for s in shapes[:-1]:
            out_ch = out_ch // 2
            self.blocks.append(
                GenBlock(in_ch, out_ch, s, is_self_attention, is_conditional)
            )
            in_ch = out_ch + 8 if is_conditional else out_ch

        out_ch = out_dim
        self.output = nn.Sequential(
            Resize(shapes[-1]),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.Softmax2d(),
        )

    def forward(self, z, label=None):
        x = self.preprocess(z)
        d, h, w = self.init_shape
        x = x.view(-1, d, h, w)
        for b in self.blocks:
            x = b(x, label)
        x = self.output(x)
        return x


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
    ):
        super(Discriminator, self).__init__()
        self.is_self_attention = is_self_attention
        self.is_minibatch_std = is_minibatch_std
        self.is_spectral_norm = is_spectral_norm

        self.preprocess = nn.Sequential(
            nn.Conv2d(in_ch, filters, 3, stride=1, padding=1), nn.LeakyReLU(True)
        )

        self.blocks = nn.ModuleList()
        in_ch = filters
        out_ch = in_ch * 2
        for s in shapes[1:]:
            self.blocks.append(
                DisBlock(in_ch, out_ch, s, is_self_attention, is_conditional)
            )
            in_ch = out_ch + 8 if is_conditional else out_ch
            out_ch = out_ch * 2

        if self.is_minibatch_std:
            self.minibatch_std = MiniBatchStd()
        # self.postprocess = nn.AdaptiveAvgPool2d(1)
        self.postprocess = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1), nn.AdaptiveAvgPool2d(1)
        )
        self.output = nn.Linear(in_ch + int(is_minibatch_std), 1)

    def forward(self, x, label=None):
        x = self.preprocess(x)
        for b in self.blocks:
            x = b(x, label)
        if self.is_minibatch_std:
            x = self.minibatch_std(x)
        x = self.postprocess(x)
        x = x.view(x.size(0), -1)
        out = self.output(x)
        return out


if __name__ == "__main__":
    model_shapes = [(3, 4), (6, 8), (12, 16)]
    generator = Generator(8, model_shapes, (128,)).to("cuda:0")
    discriminator = Discriminator(8, model_shapes[::-1], filters=32).to("cuda:0")
    batch_size = 64
    summary(
        generator,
        (
            batch_size,
            128,
        ),
    )
    summary(discriminator, (batch_size, 8, 12, 16))
