import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Opt(object):
    """
    This class defines the global optimization settings.
    """
    dim = 10
    n_epochs = 200
    batch_size = dim*dim
    lr = 0.0002
    b1 = 0.5
    b2 = 0.999
    n_cpu = 1
    latent_dim = 100
    img_size = 28
    channels = 1
    sample_interval = 400


class Generator(nn.Module):
    def __init__(self, img_shape, opt):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.opt = opt
        self.img_shape = img_shape
        self.model = nn.Sequential(
            *block(self.opt.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


def calculate_jsd(p, q):
    compare = torch.div(torch.add(p, q), 2)

    temp_1 = 0.5 * F.kl_div(p, compare)
    temp_2 = 0.5 * F.kl_div(q, compare)

    return temp_1 + temp_2

