import os
os.environ['WANDB_DIR'] = '/cluster/scratch/jonatank/wandb'

import torch
import torch.utils.data as data
import torch.optim as optim
from torchvision.utils import make_grid
from torchvision import transforms
from tensorboardX import SummaryWriter
import argparse
import yaml
from datetime import datetime
import numpy as np
from torch.autograd import Variable
from dataloader import patch_data
from vae import VAE, test, train
from utils import normalize_tensor
import glob

if __name__ == "__main__":
    # Params init
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('--modality', type=str, default='T2')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr_rate', type=float, default=1e-4)
    parser.add_argument('--zdims', type=int, default=60)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--use_KLannealing', type=int, default=0)
    parser.add_argument('--train_data', type=str, default='/cluster/work/cvl/jonatank/fastMRI_T2/train/')
    parser.add_argument('--train_coildata', type=str, default='/cluster/work/cvl/jonatank/est_coilmaps_train/')
    parser.add_argument('--val_data', type=str, default='/cluster/work/cvl/jonatank/fastMRI_T2/validation/')
    parser.add_argument('--val_coildata', type=str, default='/cluster/work/cvl/jonatank/est_coilmaps_val/')
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--img_size', type=int, default=28)
    parser.add_argument('--log_dir', type=str, default="/cluster/scratch/jonatank/logs/ddp/vae/")

    args = parser.parse_args()
    modality = args.modality  # 'FLAIR' 'T1_' 'T1POST' 'T1PRE' 'T2'
    batch_size = args.batch_size
    lr_rate = args.lr_rate
    zdims = args.zdims
    beta = args.beta
    use_KLannealing = bool(args.use_KLannealing)
    rss = 'True' # True #True

    print('Modality: ', modality)

    datapath_train = args.train_data # 
    datapath_val =  args.val_data #
    coil_path_train = args.train_coildata #
    coil_path_val = args.val_coildata # 

    epochs = args.epochs
    img_size = args.img_size
    log_freq = 100
    name = str(modality + '-' + datetime.now().strftime("%Y%m%d-%H%M%S"))
    log_dir =  args.log_dir + name
    os.mkdir(log_dir)
    print(log_dir)
    # Cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device: ' + str(device))

    # Load data
    dataset_MRI = patch_data(datapath_train, coil_path_train, modality=modality, crop=True, noiseinvstd=100, valid=0, rss=rss)
    data_loader  = data.DataLoader(dataset_MRI, batch_size=batch_size, shuffle=True, num_workers=124) #32
 
    dataset_MRI_valid = patch_data(datapath_val, coil_path_val, modality=modality, crop=True, noiseinvstd=0, num_subj=100, valid=1, rss=rss) 
    data_loader_valid  = data.DataLoader(dataset_MRI_valid, batch_size=batch_size, shuffle=False, num_workers=124) #32

    # Create model
    vae_model = VAE(zdims)
    vae_model.to(device)

    #Init logging
    import wandb 
    wandb.login()
    wandb.init(project='PriorMRI', name=name ,config={
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_rate": lr_rate,
        "zdims": zdims,
        "beta": beta,
        "KLanneal": use_KLannealing
    })
    wandb.watch(vae_model)

    print("epochs ", epochs,
        " batch_size ", batch_size,
        " lr_rate ", lr_rate,
        " zdims ", zdims,
        " beta ", beta,
        "KLanneal", use_KLannealing
    )

    print('Rss: ', rss)
    #wandb = None

    # Init Optimizer Adam
    optimizer = optim.Adam(vae_model.parameters(), lr=lr_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)

    # Start training
    print('Start training:')
    for epoch in range(1, epochs + 1): 
        print(epoch)
        # Use Kl annealing 
        if use_KLannealing:
            if (epoch - 1) % 100 == 0:
                beta = 0.5
            
            beta = min(5, beta)
            beta = beta + 1. / 10

        train(epoch, vae_model, data_loader, img_size, batch_size, optimizer, wandb, beta, device)

        if epoch % 10 == 0:
            test(epoch, vae_model, data_loader_valid, img_size, batch_size, optimizer, wandb, device)

            mu_sampler, sigma_sample = vae_model.sample(64)
            mu_sampler = transforms.ToPILImage()(normalize_tensor(make_grid(mu_sampler.view(64,1,img_size,img_size).detach().cpu())))
            sigma_sample = transforms.ToPILImage()(normalize_tensor(make_grid(sigma_sample.view(64,1,img_size,img_size).detach().cpu())))

            wandb.log({"Sampled Mu": [wandb.Image(mu_sampler, caption="")]})
            wandb.log({"Sampled Prec": [wandb.Image(sigma_sample, caption="")]})
            
        if epoch % 50 == 0:
            path = log_dir + '/' + str(epoch) + '.pth'
            torch.save(vae_model, path)

        scheduler.step()


