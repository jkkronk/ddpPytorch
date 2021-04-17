import torch
import torch.fft
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import List, Optional
import torch.fft
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from typing import Union

class rmsprop():
    def __init__(self, eta=1e-4, gamma=0.9, epsilon=1e-8):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eta = eta
        self.moment = 0

    def update(self, t, w, w_grad):
        self.moment = self.gamma*self.moment + (1-self.gamma)*(w_grad)**2
        lr_a = self.eta/torch.sqrt(self.moment + self.epsilon)

        w = w - w_grad*lr_a
        return w

class adamOptim():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta

    def update(self, t, w, w_grad):
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*w_grad
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(w_grad**2)

        m_cap = self.m_dw/(1-self.beta1**t)
        v_cap = self.v_dw/(1-self.beta2**t)

        w = w - self.eta*m_cap/(torch.sqrt(v_cap)+self.epsilon)
        return w

class momentumGrad():
    def __init__(self, eta=1e-4, beta=0.5):
        self.moment = 0
        self.beta = beta
        self.eta = eta

    def update(self, t, w, w_grad):
        if t > 1:
            self.moment = self.beta * self.moment + (1-self.beta) * w_grad
        else:
            self.moment = w_grad

        ## update weights and biases
        w = w - self.eta * self.moment
        return w

### Complex functions
def center_crop_pytorch(data):
    """
    Apply a center crop to the input real image or batch of real images.
    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.
    Returns:
        The center cropped image.
    """
    device = data.device
    if data.shape[0] < 320:
        data_p = torch.zeros((320,data.shape[1]), dtype=torch.cfloat, device=device) #+ np.random.normal(loc=0, scale=1/10, size=(320,data.shape[1]))
        data_p[:data.shape[0]] = data
        data = data_p

    if data.shape[1] < 320:
        data_p = torch.zeros((data.shape[0], 320), dtype=torch.cfloat, device=device) #+ np.random.normal(loc=0, scale=1/10, size=(data.shape[0], 320))
        data_p[:,:data.shape[1]] = data
        data = data_p

    w_from = (data.shape[0] // 2) - 160
    w_to = w_from + 320

    h_from = (data.shape[1]  // 2) - 160
    h_to = h_from + 320

    ret = data[w_from:w_to, h_from:h_to]

    return ret, w_from, w_to, h_from, h_to

def create_patches(img, parfact, patch_sz):
    device = img.device
    # Create patches and batch them with parfact
    stride = patch_sz//2
    # #H_pad = (H%stride)//2
    # #W_pad = (W%stride)//2
    # #print(H_pad, W_pad)
    # #img = F.pad(img, (H_pad, H_pad, W_pad, W_pad), mode='constant', value=0.4) # Pad to be patched by patch size
    patches = img.unfold(0, patch_sz, stride).unfold(1, patch_sz, stride)
    patches = patches.reshape(-1, *(patches.size()[2:])) # [nb_patches_h, nb_patches_w, ps, ps] --> [num_patches, ps, ps]
    
    # Create batches of parfact
    num_patches = patches.shape[0]
    inds = int(np.ceil(num_patches / parfact) * parfact)
    ext_patches = torch.zeros((inds, patches.shape[1], patches.shape[2]), dtype=img.dtype, device=device)
    ext_patches[:num_patches] = patches
    return ext_patches, num_patches

def stitch_patches(patches, patch_sz, H, W):
    device = patches.device
    # Undo create_patches()
    stride = patch_sz//2
    patches = patches.contiguous().view(1, 1, -1, patch_sz*patch_sz)
    patches = patches.permute(0, 1, 3, 2) 
    patches = patches.contiguous().view(1, 1*patch_sz*patch_sz, -1)

    # Undo patch operation
    divisor_tmp = torch.ones((H,W)).type(torch.float).to(device).unfold(0, patch_sz, stride).unfold(1, patch_sz, stride)
    divisor_tmp = divisor_tmp.reshape(-1, patch_sz, patch_sz) # [nb_patches_h, nb_patches_w, ps, ps] --> [num_patches, ps, ps]
    divisor_tmp = divisor_tmp.contiguous().view(1, 1, -1, patch_sz*patch_sz)
    divisor_tmp = divisor_tmp.permute(0, 1, 3, 2) 
    divisor_tmp = divisor_tmp.contiguous().view(1, 1*patch_sz*patch_sz, -1)
    divisor = F.fold(divisor_tmp,output_size=(H, W) , kernel_size=patch_sz, stride=stride)
    divisor[divisor==0] = 1

    img = (F.fold(patches, output_size=(H, W) , kernel_size=patch_sz, stride=stride) / divisor).squeeze(0).squeeze(0) # (H, W) 
    return img

## PYTORCH SUPPORT (SOME IMPLEMENTATIONS FROM fastMRI)
def percentile(t: torch.tensor, q: float) -> Union[int, float]:
    """
    Return the ``q``-th percentile of the flattened input tensor's data.
    
    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.
       
    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result
    
def normalize_tensor(input_tens):
    i_max = input_tens.max()
    i_min = input_tens.min()
    return (input_tens-i_min)/(i_max-i_min)

def act_func(x):
    return 1 - torch.exp(-(x**2))

def rss_from_rec(x, coilmaps):
    ksp_tmp = fftshift(torch.fft.fftn(coilmaps * x.unsqueeze(0), dim=(1, 2)), dim=(1, 2)) 
    return rss_pytorch(ksp_tmp)

def rss_pytorch(x):
    # K-space data to rss
    # inp: [nx, ny], [nx, ny]
    # out: [nx, ny]
    img_tmp = torch.fft.ifftn(ifftshift(x, dim=(1, 2)), dim=(1, 2))
    return torch.sqrt(torch.sum(torch.square(torch.abs(img_tmp)), axis=0))

def FT_pytorch(x, coilmaps):
    # inp: [nx, ny]
    # out: [nx, ny, ns]
    return fftshift(torch.fft.fftn(coilmaps * x.unsqueeze(0).repeat(coilmaps.shape[0], 1, 1), dim=(1,2)), dim=(1,2))

def tFT_pytorch(x, coilmaps):
    # inp: [nx, ny, ns]
    # out: [nx, ny]
    temp = torch.fft.ifftn(ifftshift(x, dim=(1,2)), dim=(1,2))

    temp_scoil = torch.sum(temp * torch.conj(coilmaps), axis=0)
    temp_scoil = temp_scoil / (torch.sum(coilmaps * torch.conj(coilmaps), axis=0))

    return temp_scoil

def UFT_pytorch(x, uspat, coilmaps):
    # inp: [nx, ny], [nx, ny]
    # out: [nx, ny, ns]

    return uspat.unsqueeze(0).repeat(coilmaps.shape[0], 1, 1) * FT_pytorch(x, coilmaps)

def tUFT_pytorch(x, uspat, coilmaps):
    # inp: [nx, ny], [nx, ny]
    # out: [nx, ny]

    return tFT_pytorch(uspat.unsqueeze(0).repeat(coilmaps.shape[0], 1, 1) * x, coilmaps)    

# Helper functions
def roll_one_dim(x: torch.Tensor, shift: int, dim: int) -> torch.Tensor:
    """
    Similar to roll but for only one dim.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    shift = shift % x.size(dim)
    if shift == 0:
        return x

    left = x.narrow(dim, 0, x.size(dim) - shift)
    right = x.narrow(dim, x.size(dim) - shift, shift)

    return torch.cat((right, left), dim=dim)


def roll(
    x: torch.Tensor,
    shift: List[int],
    dim: List[int],
) -> torch.Tensor:
    """
    Similar to np.roll but applies to PyTorch Tensors.

    Args:
        x: A PyTorch tensor.
        shift: Amount to roll.
        dim: Which dimension to roll.

    Returns:
        Rolled version of x.
    """
    if len(shift) != len(dim):
        raise ValueError("len(shift) must match len(dim)")

    for (s, d) in zip(shift, dim):
        x = roll_one_dim(x, s, d)

    return x


def fftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.fftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to fftshift.

    Returns:
        fftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = x.shape[dim_num] // 2

    return roll(x, shift, dim)


def ifftshift(x: torch.Tensor, dim: Optional[List[int]] = None) -> torch.Tensor:
    """
    Similar to np.fft.ifftshift but applies to PyTorch Tensors

    Args:
        x: A PyTorch tensor.
        dim: Which dimension to ifftshift.

    Returns:
        ifftshifted version of x.
    """
    if dim is None:
        # this weird code is necessary for toch.jit.script typing
        dim = [0] * (x.dim())
        for i in range(1, x.dim()):
            dim[i] = i

    # also necessary for torch.jit.script
    shift = [0] * len(dim)
    for i, dim_num in enumerate(dim):
        shift[i] = (x.shape[dim_num] + 1) // 2

    return roll(x, shift, dim)


## NUMPY SUPPORT
def min_max_aclines(ksp, N=10):
    """
    calculates the mean and std of ksp within N aclines

    For horizontal sampling 
    """
    C,H,W = ksp.shape
    ksp[:, :, :(W//2-N//2)] = 0
    ksp[:, :, (W//2+N//2):] = 0
    data_img = np.fft.ifft2(np.fft.ifftshift(ksp,axes=(1, 2)), axes=(1, 2))
    data_img = np.sqrt(np.sum(np.square(np.abs(data_img)), axis=0))
    return np.min(data_img), np.max(data_img)

def mean_std_aclines_pytorch(ksp, N=10):
    """
    calculates the mean and std of ksp within N aclines

    For horizontal sampling 
    """
    C,H,W = ksp.shape
    ksp[:, :, :(W//2-N//2)] = 0
    ksp[:, :, (W//2+N//2):] = 0
    data_img = torch.fft.ifftn(ifftshift(ksp, dim=(1, 2)), dim=(1, 2))
    data_img = torch.sqrt(torch.sum(torch.square(torch.abs(data_img)), axis=0))
    return torch.mean(data_img), torch.std(data_img)

def mean_std_aclines(ksp, N=10):
    """
    calculates the mean and std of ksp within N aclines

    For horizontal sampling 
    """
    C,H,W = ksp.shape
    ksp[:, :, :(W//2-N//2)] = 0
    ksp[:, :, (W//2+N//2):] = 0
    data_img = np.fft.ifft2(np.fft.ifftshift(ksp,axes=(1, 2)), axes=(1, 2))
    data_img = np.sqrt(np.sum(np.square(np.abs(data_img)), axis=0))
    return np.mean(data_img), np.std(data_img)


def center_crop(data):
    """
    Apply a center crop to the input real image or batch of real images.
    Args:
        data: The input tensor to be center cropped. It should
            have at least 2 dimensions and the cropping is applied along the
            last two dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.
    Returns:
        The center cropped image.
    """
    if data.shape[0] < 320:
        data_p = np.zeros((320,data.shape[1]), dtype=complex) #+ np.random.normal(loc=0, scale=1/10, size=(320,data.shape[1]))
        data_p[:data.shape[0]] = data
        data = data_p

    if data.shape[1] < 320:
        data_p = np.zeros((data.shape[0], 320), dtype=complex) #+ np.random.normal(loc=0, scale=1/10, size=(data.shape[0], 320))
        data_p[:,:data.shape[1]] = data
        data = data_p

    w_from = (data.shape[0] // 2) - 160
    w_to = w_from + 320

    h_from = (data.shape[1]  // 2) - 160
    h_to = h_from + 320

    ret = data[w_from:w_to, h_from:h_to]

    return ret

## METRICS

def nmse(pred, gt):
    return np.linalg.norm(pred.flatten() - gt.flatten()) ** 2 / np.linalg.norm(gt.flatten()) ** 2

def ssim(pred,gt):
    return structural_similarity(pred, gt, multichannel=True, data_range=(gt).max(), win_size=7)

def psnr(pred, gt):
    """ Compute Peak Signal to Noise Ratio metric (PSNR) """
    return peak_signal_noise_ratio(gt, pred, data_range=gt.max())



## FOR PHASE PROJECTION 
def tv_proj(phs, mu=0.125, lmb=2, IT=225):
    # Total variation based projection

    phs = fb_tv_proj(phs, mu=mu, lmb=lmb, IT=IT)

    return phs

def fb_tv_proj(im, u0=0, mu=0.125, lmb=1, IT=15):
    sz = im.shape
    us = np.zeros((2, sz[0], sz[1], IT))
    us[:, :, :, 0] = u0

    for it in range(IT - 1):
        # grad descent step:
        tmp1 = im - _fdivg(us[:, :, :, it])
        tmp2 = mu * _fgrad(tmp1)

        tmp3 = us[:, :, :, it] - tmp2

        # thresholding step:
        us[:, :, :, it + 1] = tmp3 - _f_st(tmp3, lmb=lmb)

        # endfor

    return im - _fdivg(us[:, :, :, it + 1])

def _fdivg(im):
    # divergence operator with 1st order finite differences
    imr_x = np.roll(np.squeeze(im[0, :, :]), shift=1, axis=0)
    imr_y = np.roll(np.squeeze(im[1, :, :]), shift=1, axis=1)
    grd_x = np.squeeze(im[0, :, :]) - imr_x
    grd_y = np.squeeze(im[1, :, :]) - imr_y

    return grd_x + grd_y

def _fgrad(im):
    # gradient operation with 1st order finite differences
    imr_x = np.roll(im, shift=-1, axis=0)
    imr_y = np.roll(im, shift=-1, axis=1)
    grd_x = imr_x - im
    grd_y = imr_y - im

    return np.array((grd_x, grd_y))

def _f_st(u, lmb):
    # soft thresholding

    uabs = np.squeeze(np.sqrt(np.sum(u * np.conjugate(u), axis=0)))

    tmp = 1 - lmb / uabs
    tmp[np.abs(tmp) < 0] = 0

    uu = u * np.tile(tmp[np.newaxis, :, :], [u.shape[0], 1, 1])

    return uu

def reg2_proj(usph, imsizer, imrizec, niter=100, alpha=0.05):
    # A smoothness based based projection. Regularization method 2 from
    # "Separate Magnitude and Phase Regularization via Compressed Sensing",  Feng Zhao et al, IEEE TMI, 2012

    usph = usph + np.pi

    ims = np.zeros((imsizer, imrizec, niter))
    ims[:, :, 0] = usph.copy()
    for ix in range(niter - 1):
        proj = ims[:, :, ix] - 2 * alpha * np.real(
            1j * np.exp(-1j * ims[:, :, ix]) * _fdivg(_fgrad(np.exp(1j * ims[:, :, ix]))))
        #print('iter {} proj2: {}'.format(ix, proj))
        ims[:, :, ix + 1] = proj.copy()

    return ims[:, :, -1] - np.pi