import torch
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np


class AdaptiveInstanceNormalization(nn.Module):

    def __init__(self, p=0.5, eps=1e-6):
        super(AdaptiveInstanceNormalization, self).__init__()
        self.eps = eps
        self.p = p
        self.factor = 1.0

    def _reparameterize(self, mu, std):
        
        epsilon = torch.randn_like(std) * self.factor
        return mu + epsilon * std

    def sqrtvar(self, x):
        t = (x.var(dim=0, keepdim=True) + self.eps).sqrt()
        t = t.repeat(x.shape[0], 1)
        return t

    def forward(self, x_cont, x_style=None):
        if x_style is not None:
            assert (x_cont.size()[:2] == x_style.size()[:2])
            style_mean, style_std = calc_mean_std(x_style)
            content_mean, content_std = calc_mean_std(x_cont)
            
            x = (x_cont - content_mean.reshape(x_cont.shape[0], x_cont.shape[1], 1, 1)) / content_std.reshape(x_cont.shape[0], x_cont.shape[1], 1, 1)

            if np.random.random() > self.p:
                denormalized_x_cont = x * style_std.reshape(x_cont.shape[0], x_cont.shape[1], 1, 1) + style_mean.reshape(x_cont.shape[0], x_cont.shape[1], 1, 1)
                return  denormalized_x_cont

            sqrtvar_mu = self.sqrtvar(style_mean)
            sqrtvar_std = self.sqrtvar(style_std)

            beta = self._reparameterize(style_mean, sqrtvar_mu)
            gamma = self._reparameterize(style_std, sqrtvar_std)

            denormalized_x_cont = x * gamma.reshape(x_cont.shape[0], x_cont.shape[1], 1, 1) + beta.reshape(x_cont.shape[0], x_cont.shape[1], 1, 1)
                        
            return denormalized_x_cont
        else:
            return x_cont


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt()
    feat_mean = feat.view(N, C, -1).mean(dim=2)
    return feat_mean, feat_std
