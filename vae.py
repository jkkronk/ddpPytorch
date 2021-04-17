import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
from utils import normalize_tensor

class VAE(nn.Module):
    def __init__(self, ZDIMS):
        super(VAE, self).__init__()

        self.ZDIMS = ZDIMS
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mc_samp = False
        self.nsamples = 50
        self.tanh = nn.Tanh()

        # Encoder blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1,
                      stride=1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1,
                      stride=1),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1,
                      stride=1),
            nn.ReLU()
        )

        # Fully connected to Latent variables
        self.fc11 = nn.Linear(in_features=64 * 28 * 28, out_features=ZDIMS)
        self.fc12 = nn.Linear(in_features=64 * 28 * 28, out_features=ZDIMS)

        # Decoder blocks
        self.fc1_t = nn.Sequential(
            nn.Linear(in_features=ZDIMS, out_features=48 * 28 * 28),
            nn.ReLU()
        )

        self.conv_t1 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=48, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.conv_t2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=90, kernel_size=3, padding=1,
                      stride=1),
            nn.ReLU()
        )

        self.conv_t3 = nn.Sequential(
            nn.Conv2d(in_channels=90, out_channels=90, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

        self.conv_t41 = nn.Conv2d(in_channels=90, out_channels=1, kernel_size=3, padding=1, stride=1)

        self.conv_t42 = nn.Conv2d(in_channels=90, out_channels=1, kernel_size=3, padding=1, stride=1)

    def encode(self, x: Variable) -> (Variable, Variable):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, 64 * 28 * 28)
        mu_z = self.fc11(x)
        logvar_z = self.fc12(x)        
        return mu_z, logvar_z

    def reparameterize(self, mu: Variable, logvar: Variable) -> list:
        if self.mc_samp:
            sample_z = []
            for _ in range(self.nsamples):
                std = torch.mul(0.5, logvar).exp()
                eps = torch.empty(std.shape, device=self.device).normal_(mean=0,std=1) #torch.randn(std.shape, device=self.device)
                sample_z.append(torch.mul(eps, std).add_(mu))
            return sample_z
        else:
            std = torch.mul(0.5, logvar).exp()
            eps = torch.empty(std.shape, device=self.device).normal_(mean=0,std=1)
            return torch.mul(eps, std).add_(mu)

    def decode(self, z: Variable) -> (Variable, Variable):
        x = self.fc1_t(z)
        x = x.view(-1, 48, 28, 28)
        deconv1 = self.conv_t1(x)
        deconv2 = self.conv_t2(deconv1)
        deconv3 = self.conv_t3(deconv2)
        mu_x = self.conv_t41(deconv3)
        inv_prec = self.conv_t42(deconv3) 
        return mu_x.reshape(mu_x.shape[0], 784), inv_prec.reshape(inv_prec.shape[0], 784)

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.reshape(x.shape[0], 784))
        z = self.reparameterize(mu, logvar)
        if self.mc_samp:
            return [self.decode(zi) for zi in z], mu, logvar, z
        else:
            return self.decode(z), mu, logvar, z

    def sample(self, numb):
        sample = torch.empty(numb, self.ZDIMS, device=self.device).normal_(mean=0,std=1) 
        mu, inv_sigma2_x = self.decode(sample)
        return mu, inv_sigma2_x

    def loss_function(self, recon_x_batch, x, mu, logvar, BATCH_SIZE) -> Variable:
        x = x.reshape(x.shape[0], 784)
        mu_x, log_sigma = recon_x_batch
        
        GLL = 0.5 * torch.sum(torch.pow((x - mu_x),2) / (log_sigma.exp()) + log_sigma, 1) # Gaussian Loss
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1) # Kl divergence loss

        return GLL, KLD

    def grad(self, x_in, nsamples):
        # Returns gradient of ELBO(x)
        self.mc_samp = True
        self.nsamples = nsamples
        self.eval()

        parfact, sz = x_in.shape
        x = x_in.unsqueeze(1)
        img_ano = nn.Parameter(x.clone(), requires_grad=True)

        recon_batch, mu, logvar, z = self.forward(img_ano)
        std = torch.mul(0.5, logvar).exp_()

        grad_l = torch.zeros((nsamples, parfact, sz)).to(self.device).float()
        inv_prec_all = torch.zeros((nsamples,parfact, sz)).to(self.device).float()
        for ix, recon_x in enumerate(recon_batch):
            mu_x, inv_prec = recon_x
            #print('mu', mu_x.min(), 'inv', inv_prec.min(), mu.min(), std.min())
            inv_prec_all[ix] = mu_x.data
            GL = -0.5 * torch.sum(torch.pow((img_ano[:,0,:] - mu_x),2) / (inv_prec.exp()) + inv_prec, 1)
            KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
            ELBO = torch.sum(GL + KLD)
            ELBO.backward(retain_graph=True)
  
            grad_l[ix] = img_ano.grad.data.detach().clone()[:,0]

            img_ano.grad.data.zero_()

        self.mc_samp = False
        return grad_l, inv_prec_all

    def ELBO(self, x_in, nsamples):
        # returns ELBO(X)
        self.mc_samp = True
        self.nsamples = nsamples
        self.eval()
        parfact, sz = x_in.shape
        x = x_in.unsqueeze(1)
        img_ano = nn.Parameter(x.clone(), requires_grad=True)

        recon_batch, mu, logvar, z = self.forward(img_ano)
        std = torch.mul(0.5, logvar).exp()
        ELBO = 0
        for ix, recon_x in enumerate(recon_batch):
            mu_x, inv_prec = recon_x
            GL = -0.5 * torch.sum(torch.pow((img_ano[:,0,:] - mu_x),2) / (inv_prec.exp()) + inv_prec, 1)
            KLD = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
            ELBO += torch.sum(GL + KLD).data.detach()

        self.mc_samp = False
        return ELBO


def train(epoch, model, train_loader, patchsize, batch_size, optimizer, writer, beta, device):
    model.train()
    sum_GLL = 0
    sum_KLD = 0   
    sum_loss = 0 

    for ix, batch in enumerate(train_loader):
        # In order to combine all patches from all samples we flatten first two dimentions and reshape.
        patches = torch.abs(batch.view(batch.shape[0]*batch.shape[1], 1, patchsize, patchsize)).to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, _ = model(patches.float())
        GLL, KLD = model.loss_function(recon_batch, patches, mu, logvar, batch_size)
        loss = torch.mean(GLL + beta * KLD)

        sum_loss += loss.item()
        sum_GLL += torch.mean(GLL).item()
        sum_KLD += torch.mean(KLD).item()
        loss.backward()

        optimizer.step()

    writer.log({"Loss":sum_loss/ix, "GLL Loss":sum_GLL/ix, "KLD Loss":sum_KLD/ix})
    print('====> batch: {} Average loss: {:.4f}'.format(epoch, sum_loss/ix))

def test(epoch, model, train_loader, patchsize, batch_size, optimizer, writer, device):
    model.eval()
    GLL_loss = 0
    KLD_loss = 0   
    test_loss = 0   
 
    for ix, batch in enumerate(train_loader):
        patches = torch.abs(batch.view(batch.shape[0]*batch.shape[1], 1, patchsize, patchsize)).to(device)
        patches.requires_grad = False
        optimizer.zero_grad()
        recon_batch, mu, logvar, z = model(patches.float())

        GLL, KLD = model.loss_function(recon_batch, patches, mu, logvar, batch_size)
        loss = torch.mean(GLL + KLD)

        loss.backward()
        test_loss += loss.item()
        GLL_loss += torch.mean(GLL).item()
        KLD_loss += torch.mean(KLD).item()

    mu_batch, sigma_batch = recon_batch
    sh1 = patches.shape[0]
    perm = torch.randperm(sh1)
    idx = perm[:64]
    patches = transforms.ToPILImage()(make_grid(normalize_tensor(patches.view(sh1, 1, 28, 28)[idx])))
    mu_batch = transforms.ToPILImage()(make_grid(normalize_tensor(mu_batch.view(sh1, 1, 28, 28)[idx])))
    sigma_batch = transforms.ToPILImage()(make_grid(normalize_tensor(sigma_batch.view(sh1, 1, 28, 28)[idx])))
    
    writer.log({"Input Img": [writer.Image(patches, caption="")]})
    writer.log({"Mu Img": [writer.Image(mu_batch, caption="")]})
    writer.log({"Prec Img": [writer.Image(sigma_batch, caption="")]})

    writer.log({"Loss Valid":test_loss / ix, "GLL Loss Valid":GLL_loss / ix, "KLD Loss Valid":KLD_loss / ix})
    print('====> TEST batch: {} Average loss: {:.4f}'.format(ix, test_loss / ix))



