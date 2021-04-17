import torch
import numpy as np 

def tv_proj(phs, mu=0.125, lmb=2, IT=225):
    # Total variation based projection
    device = phs.device
    phs = fb_tv_proj(phs, mu=mu, lmb=lmb, IT=IT, device=device)

    return phs

def reg2_proj(usph, imsizer, imrizec, niter=100, alpha=0.05):
    # A smoothness based based projection. Regularization method 2 from
    # "Separate Magnitude and Phase Regularization via Compressed Sensing",  Feng Zhao et al, IEEE TMI, 2012
    device = usph.device

    ims = torch.zeros((imsizer, imrizec, niter), device=device)
    ims[:, :, 0] = usph.clone() + np.pi
    for ix in range(niter - 1):
        ims[:, :, ix + 1] = ims[:, :, ix] - 2 * alpha * torch.real(1j * torch.exp(-1j * ims[:, :, ix]) * _fdivg(_fgrad(torch.exp(1j * ims[:, :, ix]))))

    return ims[:, :, -1] - np.pi

# Helper functions 

def fb_tv_proj(im, u0=0, mu=0.125, lmb=1, IT=15, device='cpu!'):
    sz = im.shape
    us = torch.zeros((2, sz[0], sz[1], IT)).to(device)
    us[:, :, :, 0] = u0

    for it in range(IT - 1):
        # grad descent step:
        tmp1 = im - _fdivg(us[:, :, :, it])
        tmp2 = mu * _fgrad(tmp1)

        tmp3 = us[:, :, :, it] - tmp2

        # thresholding step:
        us[:, :, :, it + 1] = tmp3 - _f_st(tmp3, lmb, device)

        # endfor

    return im - _fdivg(us[:, :, :, it + 1])

def _fdivg(im):
    # divergence operator with 1st order finite differences
    imr_x = torch.roll(torch.squeeze(im[0, :, :]), shifts=1, dims=0)
    imr_y = torch.roll(torch.squeeze(im[1, :, :]), shifts=1, dims=1)
    grd_x = torch.squeeze(im[0, :, :]) - imr_x
    grd_y = torch.squeeze(im[1, :, :]) - imr_y

    return grd_x + grd_y

def _fgrad(im):
    # gradient operation with 1st order finite differences
    imr_x = torch.roll(im, shifts=-1, dims=0)
    imr_y = torch.roll(im, shifts=-1, dims=1)
    grd_x = imr_x - im
    grd_y = imr_y - im

    return torch.stack([grd_x, grd_y], dim=0) 

def tile(a, dim, n_tile, device):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
    return torch.index_select(a, dim, order_index)

def _f_st(u, lmb, device):
    # soft thresholding
    uabs = torch.squeeze(torch.sqrt(torch.sum(u * torch.conj(u), dim=0)))
    tmp = 1 - lmb / (uabs+1e-8)
    tmp[torch.abs(tmp) < 0] = 0
    uu = u * tile(tmp.unsqueeze(0), 0, u.shape[0], device)
    return uu










