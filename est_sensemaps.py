import os 
import numpy as np
import sigpy.mri as mr
import cupy
import h5py 
from random import shuffle

datapath =  '/cluster/work/cvl/jonatank/multicoil_train/'
sensemaps_path = datapath + '../est_coilmaps_train/'
print(datapath)

files = os.listdir(datapath)  # /h5_slices
datafiles = [s for s in files if 'T2' in s]

sense_files = os.listdir(sensemaps_path)  # /h5_slices

#shuffle(datafiles)

for j in range(len(datafiles)):
	subj_file = datafiles[j]

	print(subj_file)

	with h5py.File(datapath + subj_file, 'r') as fdset:
		subj = fdset['kspace'][:]

	S, C, H, W = subj.shape

	# Rotate ksp
	img_sli = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(subj, axes=(2, 3)), axes=(2, 3)), axes=(2, 3))
	subj = np.fft.fftshift(np.fft.fft2(img_sli,axes=(2, 3)), axes=(2, 3))

	for i in range(S):
		try:
			est_coilmap = np.load(sensemaps_path + subj_file + str(i) + '.npy')
		except:
			est_coilmap_gpu = mr.app.EspiritCalib(subj[i], calib_width=W//2, thresh=0.02, kernel_width=6, crop=0.01, max_iter=100, show_pbar=False, device=0).run().get()
			est_coilmap = np.fft.fftshift(est_coilmap_gpu, axes=(1, 2))

			np.save(sensemaps_path + subj_file + str(i), est_coilmap)

			del est_coilmap
			del est_coilmap_gpu

			cupy.clear_memo()
			print(j,i)