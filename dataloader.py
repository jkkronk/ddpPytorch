import h5py
import os
import sigpy.mri as mr
import cupy
import torch
from torch.utils.data import Dataset
import numpy as np
from random import shuffle
from utils import center_crop
from US_pattern import US_pattern

class patch_data(Dataset):
    def __init__(self, dirname, coil_path, noiseinvstd=0, patchsize=28, modality='T2', valid=0, crop=True, num_subj=0, rss=False):
        self.dirname = dirname
        self.noise = noiseinvstd
        self.patchsize = patchsize
        self.crop = crop
        self.rss = rss
        self.coil_path = coil_path
        
        #self.use_abs = use_abs
        self.allfiles = os.listdir(dirname)  # /h5_slices
        self.datafiles = [s for s in self.allfiles if modality in s]

        if num_subj > 0:
            shuffle(self.datafiles)
            self.datafiles = self.datafiles[:num_subj]

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, index):
        subj_file = self.datafiles[index]
        
        with h5py.File(self.dirname + subj_file, 'r') as fdset:
            h5data = fdset['kspace']  
            sh = h5data.shape
            sliceindex = np.random.randint(0, sh[0])
            ksp_sli = h5data[sliceindex]
        
        C, H, W = ksp_sli.shape
        
        # Rotate ksp
        img_sli = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(ksp_sli, axes=(1, 2)), axes=(1, 2)), axes=(1, 2))
        ksp_sli = np.fft.fftshift(np.fft.fft2(img_sli,axes=(1, 2)), axes=(1, 2))

        if self.rss == 'norm_acl':
            #mean, std = mean_std_aclines(ksp_sli, 10)
            i_min, i_max = min_max_aclines(ksp_sli, 10)
            img_sli_singlec = np.sqrt(np.sum(np.square(np.abs(img_sli)), axis=0))
            #norm_img_sli_singlec = (img_sli_singlec-mean)/std
            #print(norm_img_sli_singlec.max(), norm_img_sli_singlec.mean(), norm_img_sli_singlec.min())
            norm_img_sli_singlec = (img_sli_singlec-i_min)/(i_max-i_min)

        elif not self.rss:
            try:
                est_coilmap = np.load(self.coil_path + subj_file + str(sliceindex) + '.npy')            
            except:
                print('CANT FIND:', self.coil_path + subj_file + str(sliceindex) ,' Creating new...')
                est_coilmap_gpu = mr.app.EspiritCalib(ksp_sli, calib_width=W, thresh=0.02, kernel_width=6, crop=0.01, max_iter=100, show_pbar=False, device=-1).run()
                est_coilmap = np.fft.fftshift(est_coilmap_gpu, axes=(1, 2))
                np.save(self.coil_path + subj_file + str(sliceindex), est_coilmap)

            if np.isnan(np.min(est_coilmap)):
                print('COILMAP Contains nan. Create new...')
                est_coilmap_gpu = mr.app.EspiritCalib(ksp_sli, calib_width=W, thresh=0.02, kernel_width=6, crop=0.01, max_iter=100, show_pbar=False, device=-1).run()
                est_coilmap = np.fft.fftshift(est_coilmap_gpu, axes=(1, 2))
                np.save(self.coil_path + subj_file + str(sliceindex), est_coilmap)
                if np.isnan(np.min(est_coilmap)):
                    print('COILMAP still contains nan...')


            # Normalise 
            img_sli_singlec = np.sum(np.conjugate(est_coilmap) * img_sli, axis=0)/(np.sum(np.conjugate(est_coilmap) * est_coilmap, axis=0)+1e-6)
            norm_fac = 1 / (np.percentile(np.abs(img_sli_singlec).flatten(), 80))
            norm_img_sli_singlec = norm_fac * np.abs(img_sli_singlec) * np.exp(1j * np.angle(img_sli_singlec)) #
        else:
            img_sli_singlec = np.sqrt(np.sum(np.square(np.abs(img_sli)), axis=0))
            norm_fac = 1 / (np.percentile(np.abs(img_sli_singlec).flatten(), 80))
            norm_img_sli_singlec = norm_fac * np.abs(img_sli_singlec)  

        if self.crop:
            norm_img_sli_singlec = center_crop(norm_img_sli_singlec) + 1j * 0

        if self.noise > 0:
            norm_img_sli_singlec = (np.abs(norm_img_sli_singlec) + np.random.normal(loc=0, scale=1/self.noise, size=norm_img_sli_singlec.shape)) * np.exp(1j * np.angle(norm_img_sli_singlec))

        if np.isnan(np.min(norm_img_sli_singlec)):
            norm_img_sli_singlec = np.zeros_like(norm_img_sli_singlec)

        # Create Patches
        subj_tens = torch.from_numpy(norm_img_sli_singlec.real).type(torch.complex64) + 1j * torch.from_numpy(norm_img_sli_singlec.imag).type(torch.complex64)
        patches = subj_tens.unfold(0, self.patchsize, self.patchsize).unfold(1, self.patchsize, self.patchsize)
        patches = torch.cat(patches.unbind()) # Patches is now [Num_patches, 28,28]

        return patches

class Subject(Dataset):
    def __init__(self, subj, path='/cluster/work/cvl/jonatank/fastMRI/multicoil_val/chunk1/', R=4, rss=True):
        self.R = R
        self.path = path
        self.subj_name = subj
        self.use_rss = rss
        with h5py.File(self.path + self.subj_name, 'r') as fdset:
            self.subj = fdset['kspace'][:]  # fdset['reconstruction_rss'][:] #
            self.sh = self.subj.shape

        self.USp = US_pattern()

    def __len__(self):
        return self.sh[0]

    def __getitem__(self, index):
        ksp_sli = self.subj[index]

        # fftshift img space to correct and get rss
        img_sli = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(ksp_sli, axes=(1, 2)), axes=(1, 2)), axes=(1, 2)) * np.sqrt(self.sh[2]*self.sh[3])
        rss = np.sqrt(np.sum(np.square(np.abs(img_sli)), axis=0))
        ksp_sli = np.fft.fftshift(np.fft.fft2(img_sli,axes=(1, 2)), axes=(1, 2))

        # Apply undersampling
        us_pat, num_low_freqs = self.USp.generate_US_pattern_pytorch(ksp_sli.shape, R=self.R)
        us_ksp = ksp_sli * us_pat

        # ESPIRIT coil estimation with sigpy 
        est_coilmap = np.fft.fftshift(mr.app.EspiritCalib(us_ksp, calib_width=num_low_freqs, thresh=0.02, kernel_width=6, 
                crop=0.01, max_iter=100, show_pbar=False, device=-1).run(), axes=(1, 2))#.get() #device=0
        #cupy.clear_memo()

        # # Normalise to 95th percentile 
        if not self.use_rss:
            temp = np.fft.ifft2(np.fft.ifftshift(us_ksp, axes=(1, 2)), axes=(1, 2))
            temp_scoil = np.abs(np.sum(temp * np.conjugate(est_coilmap), axis=0) / (np.sum(est_coilmap * np.conjugate(est_coilmap), axis=0)))
            norm_fac = 1 / np.percentile(np.abs(temp_scoil).flatten(), 80)
            norm_us_ksp = us_ksp * norm_fac
        else:
            temp = np.fft.ifft2(np.fft.ifftshift(us_ksp, axes=(1, 2)), axes=(1, 2))
            temp_scoil = np.sqrt(np.sum(np.square(np.abs(temp)), axis=0))
            norm_fac = 1 / (np.percentile(np.abs(temp_scoil).flatten(), 80)) 
            norm_us_ksp = us_ksp * norm_fac #+ np.random.normal(loc=0, scale=1/100, size=norm_img_sli_singlec.shape)

        return torch.from_numpy(norm_us_ksp), torch.from_numpy(est_coilmap), torch.from_numpy(rss).float(), norm_fac, index


