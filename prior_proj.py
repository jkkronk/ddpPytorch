from utils import tUFT_pytorch, UFT_pytorch, percentile, create_patches, stitch_patches, center_crop_pytorch
import torch
import numpy as np

def dconst_grad(img, data, uspat, coilmaps):
    # DC gradient
    # inp: [nx, ny]
    # out: [nx, ny]
    return 2 * tUFT_pytorch(UFT_pytorch(img, uspat, coilmaps) - data, uspat, coilmaps)

def likelihood_grad(img, mode, nsampl, vae_model, boot_samples, patch_sz, device):
    # Likelihood gradient calculation with grad ELBO(X)
    # inp: [parfact, ps*ps]
    # out: [parfact, ps*ps]

    img_re = img.reshape(img.shape[0], patch_sz*patch_sz).float()

    if mode == 'LUDDP' or mode == 'DDP' or mode == 'JDDP':
        # Use local samples for uncertainty estimation
        grd0eval, inv_preceval = vae_model.grad_direct(img_re, nsampl)  #vae_model.grad_direct(img_re, nsampl)  
        grd0m = torch.mean(grd0eval, dim=0)#[0]
        #print(grd0m)
        grd0std = torch.std(inv_preceval, dim=0) 
    elif mode == 'GUDDP':
        # Use Bootstrap sampling for stat. computation of gradients 
        samples, inv_prec_samples = vae_model.grad(img_re, nsampl)  
        grd0eval = torch.zeros((boot_samples, samples.shape[1], samples.shape[2]), device=device)

        for idx in range(boot_samples):
            grd0eval[idx] = torch.mean(samples[torch.random.randint(nsampl, size=int(nsampl*0.40)),:,:], dim=0)

        grd0m = torch.mean(samples, dim=0)#[0]  # [parfact,784]
        grd0std = torch.std(grd0eval, dim=0)  # [parfact,784]
    else:
        print('EXIT: No VAETV mode type: ', mode)
        exit()
    
    grd0m = grd0m.reshape(grd0m.shape[0], patch_sz, patch_sz)
    grd0std = grd0std.reshape(grd0std.shape[0], patch_sz, patch_sz)
    
    return grd0m, grd0std

def prior_gradient(img, data, uspat, coilmaps, patch_sz, parfact, nsampl, vae_model, boot_samples, mode, mean, std):
    # inp: [nx*nx, 1]
    # out: [nx*nx, 1]
    device = img.device
    grd_dc = dconst_grad(img, data, uspat, coilmaps)

    ret_tensor = torch.zeros_like(img, device=device)

    # Normalize for VAE
    norm_img = (torch.abs(img)-mean)/std
    #norm_fac = 1 / (np.percentile(np.abs(img.detach().cpu().numpy()).flatten(), 95))
    #norm_img = norm_fac * torch.abs(img) 
    #print(norm_img)
    cimg, w_from, w_to, h_from, h_to = norm_img, 0, ret_tensor.shape[0], 0, ret_tensor.shape[1] #
    H,W = cimg.shape

    # Create overlapping patches with batchsize parfact
    ext_patches, num_patches = create_patches(cimg, parfact, patch_sz)
    
    # Loop over batches to get gradient
    inds = int(np.ceil(num_patches / parfact) * parfact)
    grds = torch.zeros((inds, patch_sz, patch_sz), device=device)
    grds_std = torch.zeros((inds, patch_sz, patch_sz), device=device)
    for ix in range(int(np.ceil(num_patches / parfact))):
        grds[parfact * ix:parfact * ix + parfact, :], grds_std[parfact * ix:parfact * ix + parfact, :] = likelihood_grad(ext_patches[parfact * ix:parfact * ix + parfact, :, :], mode, nsampl, vae_model, boot_samples, patch_sz, device)
    grds = grds[:num_patches]
    grds_std = grds_std[:num_patches]

    # Make matrices fit F.fold input
    img_grds = stitch_patches(grds, patch_sz, H, W)
    grds_std = stitch_patches(grds_std, patch_sz, H, W)

    ret_tensor[w_from:w_to, h_from:h_to] = img_grds
    #print(ret_tensor)
    # Return and unormalise
    return -1 * ret_tensor*std+mean, grds_std, grd_dc

def likelihood(img, nsampl, vae_model, patch_sz):
    # inp: [parfact, ps*ps]
    # out: [parfact, ps*ps]
    img = img.reshape(img.shape[0], patch_sz*patch_sz).float()
    ELBO = vae_model.ELBO(img, nsampl) # [nsampl parfact x 1]

    return torch.sum(ELBO, dim=0) #ELBO #[parfact]

def prior_value(img, data, uspat, coilmaps, patch_sz, parfact, nsampl, vae_model, mean, std):
    # inp: [nx*nx, 1]
    # out: [nx*nx, 1]
    device = img.device

    # DC Error
    diff = (UFT_pytorch(img, uspat, coilmaps) - data).flatten()
    dc_err = torch.sqrt(torch.sum(diff.real**2 + diff.imag**2)) 
    
    norm_img = (torch.abs(img)-mean)/std
    #norm_fac = 1 / (np.percentile(np.abs(img.detach().cpu().numpy()).flatten(), 95))
    #norm_img = norm_fac * torch.abs(img) 
    
    cimg, w_from, w_to, h_from, h_to = center_crop_pytorch(norm_img)
    # Create overlapping patches with batchsize parfact
    norm_fac = 1 / np.percentile(img.detach().cpu().numpy().flatten(), 95)
    img = img * norm_fac 

    # Normalize for VAE
    ext_patches, num_patches = create_patches(cimg, parfact, patch_sz)

    # Create parfact batches
    inds = int(np.ceil(num_patches / parfact) * parfact)
    ELBO = torch.zeros((inds, patch_sz, patch_sz), device=device)

    for ix in range(int(np.ceil(num_patches / parfact))):
        ELBO[parfact * ix:parfact * ix + parfact] = likelihood(ext_patches[parfact * ix:parfact * ix + parfact, :, :], nsampl, vae_model, patch_sz)

    ELBO = ELBO[:num_patches]

    return -1 * torch.sum(ELBO), dc_err

## TV
def tv_norm(x):
    """Computes the total variation norm and its gradient. From jcjohnson/cnn-vis."""
    H, W = x.shape
    x.unsqueeze_(-1)

    x_diff = x - torch.roll(x, -1, dims=1)
    y_diff = x - torch.roll(x, -1, dims=0)
    norm = torch.sqrt(x_diff ** 2 + y_diff ** 2 + 0.1 )

    dgrad_norm = 0.5 / norm
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    grad = dx_diff + dy_diff
    grad[:, 1:, :] -= dx_diff[:, :-1, :]
    grad[1:, :, :] -= dy_diff[:-1, :, :]

    grad.flatten()

    return norm, torch.reshape(grad, (H, W))
