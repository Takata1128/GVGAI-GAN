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

    def forward(self, x, label=None):
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
    def __init__(self, in_dim, input_shape, label_channels):
        super().__init__()
        self.in_dim = in_dim
        self.shape = input_shape
        self.channels = label_channels
        self.attn = Self_Attn(in_dim)
        self.embedding = nn.Linear(label_channels, label_channels)
        self.resize = nn.Upsample(input_shape)

    def forward(self, x, label):
        x = self.attn(x)
        label = label.float()
        label = torch.softmax(label, -1)
        u = self.embedding(label)
        u = u.view(-1, self.channels, 1, 1)
        u = self.resize(u)
        x = torch.cat([x, u], dim=1)
        return x


class MiniBatchStd(nn.Module):
    def forward(self, x):
        std = torch.std(x, dim=0, keepdim=True)
        mean = torch.mean(std, dim=(1, 2, 3), keepdim=True)
        n, c, h, w = x.shape
        mean = torch.ones(n, 1, h, w, dtype=x.dtype, device=x.device) * mean
        return torch.cat((x, mean), dim=1)


class Discriminator(nn.Module):
    def __init__(self, isize, nz, nc, ndf, self_attention, n_extra_layers=0):
        super(Discriminator, self).__init__()
        self.self_attention = self_attention
        self.nc = nc
        self.isize = isize
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        if self.self_attention:
            main.add_module('initial_{0}_selfattn'.format(nc), Self_Attn(nc))
            main.add_module('initial_{0}_tanh'.format(nc),
                            nn.LeakyReLU(0.2, inplace=True))  # nn.Softmax(1))    #Was TANH nn.Tanh())#

        main.add_module('initial_conv_{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial_relu_{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            if self.self_attention:
                main.add_module('pyramid_{0}_selfattn'.format(
                    out_feat), Self_Attn(in_feat))
            main.add_module('pyramid_{0}-{1}_conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))

            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final_{0}-{1}_conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, input, label=None):
        output = self.main(input)

        output = output.mean(0)
        return output.view(1)

    def summary(self, batch_size=64, device=None):
        summary(self, (batch_size, self.nc,
                       self.isize, self.isize), device=device)


class Generator(nn.Module):
    def __init__(self, isize, nz, nc, ngf, self_attention=False, n_extra_layers=0):
        super(Generator, self).__init__()
        self.self_attention = self_attention
        self.nz = nz
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        self.mods = []

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial_{0}-{1}_convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf),
                        nn.ReLU(True))

        self.mods.append(main)

        csize, cndf = 4, cngf
        while csize < isize // 2:
            main = nn.Sequential()
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf // 2),
                            nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf // 2),
                            nn.BatchNorm2d(cngf // 2))
            main.add_module('pyramid_{0}_relu'.format(cngf // 2),
                            nn.ReLU(True))
            if self.self_attention:
                main.add_module('pyramid_{0}_selfattn'.format(
                    cngf // 2), Self_Attn(cngf // 2))
            self.mods.append(main)
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main = nn.Sequential()
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cngf),
                            nn.ReLU(True))
            self.mods.append(main)

        main = nn.Sequential()
        main.add_module('final_{0}-{1}_convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final_{0}_tanh'.format(nc),
                        nn.ReLU())  # nn.Softmax(1))    #Was TANH nn.Tanh())#
        if self.self_attention:
            main.add_module('final_{0}_selfattn'.format(nc), Self_Attn(nc))
            main.add_module('final_{0}_tanh'.format(nc),
                            nn.ReLU())  # nn.Softmax(1))    #Was TANH nn.Tanh())#
        self.mods.append(main)
        self.mods = nn.ModuleList(self.mods)

    def forward(self, input, label=None):
        hidden = input.reshape(*input.shape, 1, 1)
        hiddens = []
        for module in self.mods:
            hidden = module(hidden)
            hiddens.append(hidden)
        output = hidden
        return output, hiddens

    def summary(self, batch_size=64, device=None):
        summary(self, (batch_size, self.nz), device=device)
