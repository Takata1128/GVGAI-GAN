from __future__ import annotations
import torch
import torch.nn as nn
from functools import reduce
from operator import mul
from torchinfo import summary
from torch.autograd import Variable


class SpectralNorm:
    '''
    Implementation of Spectral Normalization for PyTorch
    https://gist.github.com/rosinality/a96c559d84ef2b138e486acf27b5a56e
    '''

    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        u = getattr(module, self.name + '_u')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        if weight_mat.is_cuda:
            u = u.cuda()
        v = weight_mat.t() @ u
        v = v / v.norm()
        u = weight_mat @ v
        u = u / u.norm()
        weight_sn = weight_mat / (u.t() @ weight_mat @ v)
        weight_sn = weight_sn.view(*size)

        return weight_sn, Variable(u.data)

    @staticmethod
    def apply(module, name):
        fn = SpectralNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        input_size = weight.size(0)
        u = Variable(torch.randn(input_size, 1) * 0.1, requires_grad=False)
        setattr(module, name + '_u', u)
        setattr(module, name, fn.compute_weight(module)[0])

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn, u = self.compute_weight(module)
        setattr(module, self.name, weight_sn)
        setattr(module, self.name + '_u', u)


def spectral_norm(module, name='weight'):
    SpectralNorm.apply(module, name)

    return module


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
    def __init__(self, in_dim, shape):
        super().__init__()
        self.shape = shape
        self.attn = Self_Attn(in_dim)
        self.in_dim = in_dim
        num_embeddings = mul(*self.shape)
        self.embedding = nn.Embedding(num_embeddings, num_embeddings)

    def forward(self, x, label=None):
        x = self.attn(x)
        u = self.embedding(label)
        u = u.view(-1, label.shape[1], *self.shape)
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
                in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.upsample = nn.Sequential(nn.Upsample(
                scale_factor=2), nn.Conv2d(in_ch, out_ch, 3, 1, 1))
        if use_bn:
            self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(True)

    def forward(self, x, label=None):
        x = self.upsample(x)
        if self.use_bn:
            x = self.bn(x)
        x = self.act(x)
        return x


class DisBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_bn=False, use_sn=True):
        super().__init__()
        self.use_bn = use_bn
        self.conv = nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1)
        if use_sn:
            self.conv = spectral_norm(self.conv)
        elif use_bn:
            self.bn = nn.BatchNorm2d(out_ch)

        self.act = nn.LeakyReLU(0.2, True)

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
        filters=64,
        use_linear4z2features_g=True,
        use_self_attention=True,
        use_conditional=False,
        use_deconv_g=True
    ):
        super(Generator, self).__init__()
        self.z_size = z_shape[0]
        self.filters = filters
        self.init_shape = shapes[0]
        self.use_linear4z2features_g = use_linear4z2features_g
        self.use_self_attention = use_self_attention
        self.use_conditional = use_conditional
        self.use_deconv_g = use_deconv_g
        self.out_dim = out_dim

        if not use_linear4z2features_g:
            self.preprocess = nn.Sequential(
                nn.ConvTranspose2d(self.z_size, filters*4,
                                   self.init_shape, 1, 0, bias=False),
                nn.BatchNorm2d(filters*4),
                nn.ReLU(True)
            )
        else:
            self.preprocess = nn.Sequential(
                nn.Linear(self.z_size, mul(filters*4, mul(*self.init_shape))),
                nn.ReLU(),
            )

        # if self.use_self_attention:
        #     self.self_attn0 = Self_Attn(filters*4)

        self.block1 = GenBlock(
            filters*4, filters*2, use_deconv_g
        )

        if self.use_self_attention:
            self.self_attn1 = Self_Attn(filters*2)

        self.block2 = GenBlock(
            filters*2, filters, use_deconv_g
        )
        if self.use_conditional:
            self.self_attn2 = ConditionalSelfAttention(
                filters, shapes[-1])
        elif self.use_self_attention:
            self.self_attn2 = Self_Attn(filters)

        outconv_ch = filters + (0 if not self.use_conditional else out_dim)
        self.outconv = nn.Sequential(nn.Conv2d(outconv_ch, out_dim,
                                               kernel_size=1, stride=1), nn.ReLU())

    def forward(self, z, label=None):
        if not self.use_linear4z2features_g:
            x = z.view(-1, self.z_size, 1, 1)
            x = self.preprocess(x)
        else:
            x = self.preprocess(z)
            x = x.view(-1, self.filters*4, *self.init_shape)

        # if self.use_self_attention:
        #     x = self.self_attn0(x)
        x = self.block1(x)

        if self.use_self_attention:
            x = self.self_attn1(x)

        x = self.block2(x)
        if self.use_conditional:
            x = self.self_attn2(x, label)
        elif self.use_self_attention:
            x = self.self_attn2(x)
        x = self.outconv(x)
        return x

    def summary(self, batch_size=32):
        if self.use_conditional:
            summary(self, ((batch_size, self.z_size),
                    (batch_size, self.out_dim)), dtypes=[torch.float, torch.int])
        else:
            summary(self, (batch_size, self.z_size),)


class Discriminator(nn.Module):
    def __init__(
        self,
        in_ch,
        shapes,
        filters=64,
        use_bn=False,
        use_self_attention=True,
        use_minibatch_std=False,
        use_recon_loss=False,
        use_conditional=False,
        use_sn=True,
        use_pooling=False
    ):
        super(Discriminator, self).__init__()
        self.use_minibatch_std = use_minibatch_std
        self.use_recon_loss = use_recon_loss
        self.use_self_attention = use_self_attention
        self.use_conditional = use_conditional
        self.use_pooling = use_pooling
        self.use_sn = use_sn
        self.input_ch = in_ch
        self.input_shape = shapes[0]

        preprocess_conv = nn.Conv2d(in_ch, filters, 1, stride=1, bias=True)
        if self.use_sn:
            self.preprocess = nn.Sequential(
                spectral_norm(preprocess_conv),
                nn.LeakyReLU(0.2, True),
            )
        else:
            self.preprocess = nn.Sequential(
                preprocess_conv,
                nn.LeakyReLU(0.2, True),
            )

        if self.use_conditional:
            self.self_attn1 = ConditionalSelfAttention(
                filters, self.input_shape)
        elif self.use_self_attention:
            self.self_attn1 = Self_Attn(filters)

        block1_ch = filters + \
            (0 if not self.use_conditional else self.input_ch)

        self.block1 = DisBlock(
            block1_ch, filters*2, use_bn=use_bn, use_sn=self.use_sn)

        if self.use_self_attention:
            self.self_attn2 = Self_Attn(filters*2)

        self.block2 = DisBlock(filters*2, filters*4,
                               use_bn=use_bn, use_sn=self.use_sn)

        # if self.use_self_attention:
        #     self.self_attn3 = Self_Attn(filters*4)

        if self.use_recon_loss:
            self.decoder = Decoder(filters*4, self.input_ch)

        if self.use_minibatch_std:
            self.minibatch_std = MiniBatchStd()

        if self.use_pooling:
            self.postprocess = nn.AdaptiveAvgPool2d(1)
        else:
            if self.use_sn:
                self.postprocess = spectral_norm(
                    nn.Conv2d(filters*4 + int(self.use_minibatch_std), 1, shapes[-1], 1, 0))
            else:
                self.postprocess = nn.Conv2d(
                    filters*4 + int(self.use_minibatch_std), 1, shapes[-1], 1, 0)

    def forward(self, x, label=None):
        x = self.preprocess(x)

        if self.use_conditional:
            x = self.self_attn1(x, label)
        elif self.use_self_attention:
            x = self.self_attn1(x)

        x = self.block1(x)

        if self.use_self_attention:
            x = self.self_attn2(x)

        x = self.block2(x)

        # if self.use_self_attention:
        #     x = self.self_attn3(x)

        branch_x = x

        if self.use_minibatch_std:
            x = self.minibatch_std(x)

        x = self.postprocess(x)

        out = x.view(x.size(0), -1)

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
                    (batch_size, self.input_ch)), dtypes=[torch.float, torch.int])
        else:
            summary(self, (batch_size, self.input_ch, *self.input_shape))


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.block1 = GenBlock(in_channel, in_channel//2, use_bn=True)
        self.block2 = GenBlock(in_channel//2, in_channel//4, use_bn=True)
        self.attn = Self_Attn(in_channel//4)
        self.conv = nn.Conv2d(in_channel//4, out_channel,
                              kernel_size=1, stride=1, bias=True)
        self.softmax = nn.Softmax2d()

    def forward(self, x, label=None):
        x = self.block1(x)
        x = self.block2(x)
        x = self.attn(x)
        x = self.conv(x)
        return self.softmax(x)
