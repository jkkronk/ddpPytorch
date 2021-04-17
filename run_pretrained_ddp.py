import os, sys
os.environ['WANDB_DIR'] = '/cluster/scratch/jonatank/wandb'

import torch
import time
import argparse
import numpy as np
from reconstruction import vaerecon
import torch.utils.data as data
import wandb 
from datetime import datetime
from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils import center_crop, nmse, ssim, psnr
from dataloader import Subject

def run_inference(subj, R, mode, k, num_sampels, num_bootsamles, batch_size, num_iter, step_size, phase_step, complex_rec, use_momentum, log, device):
    # Load pretrained VAE
    
    vae_model_name = 'T2-20210415-111101/450.pth' #'T2-20210413-135556/400.pth' #'T2-20210406-111637/600.pth' #'T2-20210405-143836/400.pth' #'T2-20210406-194637/100.pth'#'T2-20210406-111637/150.pth' #'T2-20210405-143836/200.pth' #'T2-20210325-123355/500.pth' #'T2-20210325-125621/500.pth'#'T220210322-092044600.pth' #'T220210317-145444400.pth' #'AXFLAIR_abs_KL_20210226-1108436000.pth'
    vae_path = '/cluster/scratch/jonatank/logs/ddp/vae/'
    data_path = '/cluster/work/cvl/jonatank/fastMRI_T2/validation/' #'/cluster/work/cvl/jonatank/fastMRI/multicoil_val/chunk1/'
    log_path = '/cluster/scratch/jonatank/logs/ddp/restore/pytorch/'
    rss = True

    path = vae_path + vae_model_name
    vae = torch.load(path, map_location=torch.device(device))
    vae.eval()

    # Data loader setup
    subj_dataset = Subject(subj, data_path, R, rss=rss)
    subj_loader  = data.DataLoader(subj_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Run model 
    start_time = time.perf_counter()
    rec_subj = np.zeros((len(subj_loader), 320, 320))
    gt_subj = np.zeros((len(subj_loader), 320, 320))

    # Set basic parameters
    print('Subj: ', subj, ' R: ', R, ' mode: ', mode, ' k: ', k, ' num_sampels: ', num_sampels, ' num_bootsamles: ', num_bootsamles, ' batch_size: ', 
        batch_size, ' num_iter: ', num_iter, ' step_size: ', step_size, ' phase_step: ', phase_step )

    # Log
    log_path = log_path + 'R' + str(R) + '_mode' + str(k) + mode + '_reg2lmb0.01_' + datetime.now().strftime("%Y%m%d-%H%M%S")
    if log:
        import wandb 
        wandb.login()
        wandb.init(project='JDDP' + '_T2', name=vae_model_name ,config={
        "num_iter": num_iter,
        "step_size": step_size,
        "phase_step": phase_step,
        "mode": mode,
        'R': R,
        'K': k, 
        'use_momentum': use_momentum
        })
        #wandb.watch(vae)
    else:
        wandb = False

    print("num_iter", num_iter,
        " step_size ", step_size,
        " phase_step ", phase_step,
        " mode ", mode,
        ' R ', R,
        ' K ', k,
        'use_momentum', use_momentum
        )

    for batch in tqdm(subj_loader, desc="Running inference"):
        ksp, coilmaps, rss, norm_fact, num_sli = batch

        
        rec_sli = vaerecon(ksp[0], coilmaps[0], mode, vae, rss[0], log_path, device, writer=wandb, 
                norm=norm_fact.item(), nsampl=num_sampels, boot_samples=num_bootsamles, k=k, patchsize=28, 
                parfact=batch_size, num_iter=num_iter, stepsize=step_size, lmb=phase_step, use_momentum=use_momentum) 

        rec_subj[num_sli] = np.abs(center_crop(rec_sli.detach().cpu().numpy()))
        gt_subj[num_sli] = np.abs(center_crop(rss[0]))

        rmse_sli = nmse(rec_subj[num_sli], gt_subj[num_sli]) 
        ssim_sli = ssim(rec_subj[num_sli], gt_subj[num_sli])
        psnr_sli = psnr(rec_subj[num_sli], gt_subj[num_sli])
        print('Slice: ', num_sli.item(), ' RMSE: ', str(rmse_sli), ' SSIM: ', str(ssim_sli),' PSNR: ', str(psnr_sli))
        end_time = time.perf_counter()

        print(f"Elapsed time for {str(num_sli)} slices: {end_time-start_time}")

    rmse_v = nmse(recon_subj, gt_subj) 
    ssim_v = nmse(recon_subj, gt_subj)
    psnr_v = nmse(recon_subj, gt_subj)
    print('Subject Done: ', 'RMSE: ', str(rmse_sli), ' SSIM: ', str(ssim_sli),' PSNR: ', str(psnr_sli))

    pickle.dump(recon_subj, open(log_path + subj + str(k) + mode + str(restore_sense) + str(R), 'wb'))

    end_time = time.perf_counter()

    print(f"Elapsed time for {len(subj_loader)} slices: {end_time-start_time}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--subj', type=str, default='file_brain_AXT2_200_2000019.h5') #ab1 file_brain_AXFLAIR_201_6002975.h5') #much ab file_brain_AXFLAIR_201_6002899.h5') #') # file_brain_AXFLAIR_200_6002462.h5 file_brain_AXFLAIR_200_6002447.h5
    parser.add_argument('--usfact', type=int, default=4)
    parser.add_argument('--mode', type=str, default='JDDP') #DDP/JDDP/GUDDP/MUDDP
    parser.add_argument('--k', type=float, default=1)
    parser.add_argument('--num_sampels', type=int, default=75)
    parser.add_argument('--num_bootsamles', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=45)
    parser.add_argument('--num_iter', type=int, default=250)
    parser.add_argument('--step_size', type=float, default=1e-3)
    parser.add_argument('--phase_step', type=float, default=0.01)
    parser.add_argument('--complex_rec', type=int, default=0)
    parser.add_argument('--use_momentum', type=int, default=0)
    parser.add_argument('--log', type=int, default=1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Running reconstruction on Device: ', device)

    args = parser.parse_args()
    run_inference(
        args.subj,
        args.usfact,
        args.mode,
        args.k,
        args.num_sampels, 
        args.num_bootsamles, 
        args.batch_size, 
        args.num_iter, 
        args.step_size, 
        args.phase_step, 
        bool(args.complex_rec), 
        bool(args.use_momentum),
        bool(args.log), 
        device
    )
