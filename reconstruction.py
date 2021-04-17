import numpy as np
import torch
from torchvision import transforms
from utils import tUFT_pytorch, tFT_pytorch, rss_pytorch, act_func, normalize_tensor, rss_from_rec, UFT_pytorch, FT_pytorch, nmse, ssim, psnr, adamOptim, momentumGrad, rmsprop, mean_std_aclines_pytorch
from jsense import create_basis_functions, sense_estimation_ls
from prior_proj import tv_norm, prior_gradient, prior_value, dconst_grad
from phase_proj import tv_proj, reg2_proj

def vaerecon(ksp, coilmaps, mode, vae_model, gt, logdir, device, writer=False, norm=1, 
            nsampl=100, boot_samples=500, k=1, patchsize=28, parfact=25, num_iter=200, 
            stepsize=5e-4, lmb=0.01, num_priors=1, use_momentum=True):
    # Init data 
    # ===============================
    imcoils, imsizer, imsizec = ksp.shape
    ksp = ksp.to(device)
    coilmaps = coilmaps.to(device)
    vae_model = vae_model.to(device)
    uspat = (torch.abs(ksp[0]) > 0).type(torch.uint8).to(device)
    recs_gpu = tUFT_pytorch(ksp, uspat, coilmaps)
    rss = rss_pytorch(ksp)
    mean, std = mean_std_aclines_pytorch(ksp.clone(), 10)

    # Init coilmaps estimation
    # ===============================
    if mode == 'JDDP':
        # Polynomial order 
        max_basis_order = 6 
        num_coeffs = (max_basis_order + 1) ** 2

        # Create the basis functions for the sense estimation estimation
        basis_funct = create_basis_functions(imsizer, imsizec, max_basis_order, show_plot=False) 
        plot_basis = False
        if plot_basis:
            for i in range(num_coeffs):
                writer.log({"Basis funcs": [writer.Image(transforms.ToPILImage()(normalize_tensor(torch.from_numpy(basis_funct[i, :, :]))), caption="")]})

        basis_funct = torch.from_numpy(np.tile(basis_funct[np.newaxis, :, :, :], [coilmaps.shape[0], 1, 1, 1])).to(device)
        coeffs_array = sense_estimation_ls(ksp, recs_gpu, basis_funct, uspat)

        coilmaps = torch.sum(coeffs_array[:, :, np.newaxis, np.newaxis] * basis_funct, 1).to(device)

        recs_gpu = tUFT_pytorch(ksp, uspat, coilmaps)

        if writer:
            for i in range(coilmaps.shape[0]):
                writer.log({"abs Coilmaps": [writer.Image(transforms.ToPILImage()(normalize_tensor(torch.abs(coilmaps[i,:,:]))), caption="")]}, step=0)
                writer.log({"phase Coilmaps": [writer.Image(transforms.ToPILImage()(normalize_tensor(torch.angle(coilmaps[i,:,:]))), caption="")]}, step=0)
        print("Coilmaps init done")

    # Log
    if writer:
        writer.log({"Gt rss": [writer.Image(transforms.ToPILImage()(normalize_tensor(gt)), caption="")]}, step=0)
        writer.log({"Restored rss": [writer.Image(transforms.ToPILImage()(normalize_tensor(rss)), caption="")]}, step=0)
        writer.log({"Restored abs": [writer.Image(transforms.ToPILImage()(normalize_tensor(torch.abs(recs_gpu))), caption="")]}, step=0)
        writer.log({"Restored Phase": [writer.Image(transforms.ToPILImage()(normalize_tensor(torch.angle(recs_gpu))), caption="")]}, step=0)
        writer.log({"diff rss": [writer.Image(transforms.ToPILImage()(normalize_tensor((rss.detach().cpu()/norm - gt.detach().cpu()))), caption="")]}, step=0)
        ssim_v = ssim(rss[160:-160].detach().cpu().numpy()/norm, gt[160:-160].detach().cpu().numpy())
        nmse_v = nmse(rss[160:-160].detach().cpu().numpy()/norm, gt[160:-160].detach().cpu().numpy()) 
        psnr_v = psnr(rss[160:-160].detach().cpu().numpy()/norm, gt[160:-160].detach().cpu().numpy())
        print('SSIM: ', ssim_v, ' NMSE: ', nmse_v, ' PSNR: ', psnr_v)
        writer.log({"SSIM":ssim_v, "NMSE":nmse_v, "PSNR":psnr_v}, step=0)

        lik, dc = prior_value(rss , ksp, uspat, coilmaps, patchsize, parfact, nsampl, vae_model, mean, std)
        writer.log({"ELBO":lik}, step=0)
        writer.log({"DC err":dc}, step=0)

    optim = momentumGrad(eta=stepsize , beta=0.9) #rmsprop(eta=stepsize, gamma=0.9, epsilon=1e-1) #adamOptim(eta=stepsize, epsilon=1e-1) #   # # # 

    t = 1
    for it in range(0, num_iter, 2):
        print('Itr: ' ,it)

        #stepsize = stepsize * 0.9

        # now do magnitude prior projection
        # ===============================================
        # ===============================================
        for _ in range(num_priors):
            # Gradient descent of Prior
            if mode == 'TV':
                tvnorm, abstvgrad = tv_norm(torch.abs(rss))
                priorgrad = abstvgrad * recs_gpu / (torch.abs(recs_gpu)) 
                recs_gpu = recs_gpu - stepsize * priorgrad

                if writer: #and it%10 == 0:
                    writer.log({"TVgrad": [writer.Image(transforms.ToPILImage()(normalize_tensor(abstvgrad)), caption="")]}, step=it+1)
                    writer.log({"TV": [writer.Image(transforms.ToPILImage()(normalize_tensor(tvnorm)), caption="")]}, step=it+1)

            elif mode == 'DDP' or mode == 'JDDP':
                g_abs_lik, est_uncert , g_dc  = prior_gradient(rss, ksp, uspat, coilmaps, patchsize, parfact, nsampl, vae_model, boot_samples, mode, mean, std)
                priorgrad = g_abs_lik * recs_gpu / (torch.abs(recs_gpu)) 

                if it > -1:
                    if use_momentum:
                        recs_gpu = optim.update(t, w=recs_gpu, w_grad=priorgrad)
                        t = t+1
                    else:
                        recs_gpu = recs_gpu - stepsize * priorgrad

                if writer: # and it%10 == 0:
                    writer.log({"VAEgrad abs": [writer.Image(transforms.ToPILImage()(normalize_tensor(torch.abs(g_abs_lik))), caption="")]}, step=it+1)
                    writer.log({"STD": torch.mean(torch.abs(est_uncert))}, step=it+1)
                    
                    tmp1 = UFT_pytorch(recs_gpu, 1-uspat, coilmaps)
                    tmp2 = ksp * uspat.unsqueeze(0) 
                    tmp = tmp1 + tmp2 
                    rss = rss_pytorch(tmp) 
                    nmse_v = nmse((rss[160:-160].detach().cpu().numpy()/norm), gt[160:-160].detach().cpu().numpy()) 
                    ssim_v = ssim(rss[160:-160].detach().cpu().numpy()/norm, gt[160:-160].detach().cpu().numpy())
                    psnr_v = psnr(rss[160:-160].detach().cpu().numpy()/norm, gt[160:-160].detach().cpu().numpy())
                    print('SSIM: ', ssim_v, ' NMSE: ', nmse_v, ' PSNR: ', psnr_v)
                    writer.log({"SSIM":ssim_v, "NMSE":nmse_v, "PSNR":psnr_v}, step=it+1)

            elif mode == 'GUDDP' or mode == 'MUDDP':
                g_abs_lik, est_uncert , g_dc = prior_gradient(rss, ksp, uspat, coilmaps, patchsize, parfact, nsampl, vae_model, boot_samples, mode)
                tvnorm, abstvgrad = tv_norm(torch.abs(recs_gpu))

                full_grad = (1-act_func(k * est_uncert)) * g_abs_lik + act_func(k * est_uncert) * abstvgrad
                priorgrad = VAETVgrad * recs_gpu / (torch.abs(recs_gpu))
                if it > -1:
                    recs_gpu = recs_gpu - stepsize * priorgrad
                
                if writer:
                    writer.log({"Est uncert": [writer.Image(transforms.ToPILImage()(normalize_tensor(est_uncert)), caption="")]}, step=it+1)
                    writer.log({"Act Est uncert": [writer.Image(transforms.ToPILImage()(act_func(k * est_uncert)), caption="")]}, step=it+1)
                    writer.log({"VAETVgrad": [writer.Image(transforms.ToPILImage()(normalize_tensor(full_grad)), caption="")]}, step=it+1)
                    writer.log({"STD": torch.mean(torch.abs(est_uncert))})
            else:
                print("Error: Prior method does not exists.")
                exit()

        # now do phase projection
        # ===============================================
        # ===============================================
        if lmb > 0:
            tmpa = torch.abs(recs_gpu)
            tmpp = torch.angle(recs_gpu)

            # We apply phase regularization to prefer smooth phase images
            #tmpptv = reg2_proj(tmpp, imsizer, imsizec, alpha=lmb, niter=2)  # 0.1, 15

            tmpptv = tv_proj(tmpp, mu=0.125, lmb=lmb, IT=50)  # 0.1, 15

            # We combine back the phase and the magnitude
            recs_gpu = tmpa * torch.exp(1j * tmpptv)

        # now do coilmaps estimation
        # ===============================================
        # ===============================================
        if mode == 'JDDP':
            # computed on cpu since pytorch gpu can handle complex numbers...
            coeffs_array = sense_estimation_ls(ksp, recs_gpu, basis_funct, uspat)
            coilmaps = torch.sum(coeffs_array[:, :, np.newaxis, np.newaxis] * basis_funct, 1).to(device)

            if writer:
                writer.log({"abs Coilmaps": [writer.Image(transforms.ToPILImage()(normalize_tensor(torch.abs(coilmaps[0,:,:]))), caption="")]}, step=it+1)
                writer.log({"phase Coilmaps": [writer.Image(transforms.ToPILImage()(normalize_tensor(torch.angle(coilmaps[0,:,:]))), caption="")]}, step=it+1)

        # now do data consistency projection
        # ===============================================
        # ===============================================
        tmp1 = UFT_pytorch(recs_gpu, 1-uspat, coilmaps)
        tmp2 = ksp * uspat.unsqueeze(0) 
        tmp = tmp1 + tmp2 
        recs_gpu = tFT_pytorch(tmp, coilmaps)
        # recs[it + 2] = recs_gpu.detach().cpu().numpy()
        rss = rss_pytorch(tmp) 

        # now log
        # ===============================================
        # ===============================================
        nmse_v = nmse((rss[160:-160].detach().cpu().numpy()/norm), gt[160:-160].detach().cpu().numpy()) 
        ssim_v = ssim(rss[160:-160].detach().cpu().numpy()/norm, gt[160:-160].detach().cpu().numpy())
        psnr_v = psnr(rss[160:-160].detach().cpu().numpy()/norm, gt[160:-160].detach().cpu().numpy())
        print('SSIM: ', ssim_v, ' NMSE: ', nmse_v, ' PSNR: ', psnr_v)

        if writer: 
            writer.log({"SSIM":ssim_v, "NMSE":nmse_v, "PSNR":psnr_v}, step=it+1)
            writer.log({"Restored rss": [writer.Image(transforms.ToPILImage()(normalize_tensor(rss)), caption="")]}, step=it+1)
            writer.log({"Restored Phase": [writer.Image(transforms.ToPILImage()(normalize_tensor(torch.angle(recs_gpu))), caption="")]}, step=it+1)
            writer.log({"diff rss": [writer.Image(transforms.ToPILImage()(normalize_tensor((rss.detach().cpu()/norm - gt.detach().cpu()))), caption="")]}, step=it+1)
            writer.log({"Restored 1ch kspace": [writer.Image(transforms.ToPILImage()(normalize_tensor(torch.log(torch.abs(tmp[0])))), caption="")]}, step=it+1)
            lik, dc = prior_value(rss, ksp, uspat, coilmaps, patchsize, parfact, nsampl, vae_model, mean, std)
            writer.log({"ELBO":lik}, step=it+1)
            writer.log({"DC err":dc}, step=it+1)
        
    return rss/norm









