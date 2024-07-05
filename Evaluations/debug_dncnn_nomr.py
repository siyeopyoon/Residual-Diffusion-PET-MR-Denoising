import os
import numpy as np
import nibabel
import skimage.metrics as metric
from skimage.filters import threshold_multiotsu
from mahotas.features import haralick
import torch

def get_npy(path):
    niftifile = nibabel.load(path)
    volume = niftifile.get_fdata()
    volume = volume.astype(np.float32)

    #volume=(volume-np.min(volume))/(np.max(volume)-np.min(volume))
    #volume=np.clip(volume,0.0,np.max(volume))
    return volume

def min_max_norm(mtx,min_val,max_val):
    return (mtx-min_val)/(max_val-min_val)
def get_MSE(src,trg):
    MSE = np.mean((trg - src)**2)

    return MSE
def get_MAE(src,trg):
    MAE = np.mean(np.abs(trg - src))

    return MAE

def get_PSNR(src,trg,min_val,max_val):
    psrc=min_max_norm(src,min_val,max_val)
    ptrg=min_max_norm(trg,min_val,max_val)
    psnr=metric.peak_signal_noise_ratio(psrc,ptrg)

    return psnr
def get_SSIM(src,trg,min_val,max_val):
    psrc=min_max_norm(src,min_val,max_val)
    ptrg=min_max_norm(trg,min_val,max_val)
    ssim=metric.structural_similarity(psrc,ptrg,data_range=1.0)

    return ssim


def get_haralic_slice(img1,img2):
# range: (0, 1)
    img1[img1<0] = 0
    img2[img2<0] = 0
    img1 = (255*img1).astype(int)
    img2 = (255 * img2).astype(int)
    img1_feature = haralick(img1).mean(axis=0)
    img2_feature = haralick(img2).mean(axis=0)
    return np.abs(img1_feature-img2_feature)#np.linalg.norm()


def get_haralic (src,trg,min_val,max_val):
    psrc=min_max_norm(src,min_val,max_val)
    ptrg=min_max_norm(trg,min_val,max_val)
    z_length=psrc.shape[-1]

    accum_score=np.zeros(13)
    for zslice in range (z_length):
        accum_score=accum_score+ get_haralic_slice(psrc[:,:,zslice],        ptrg[:,:,zslice])

    accum_score=np.sum(abs(accum_score/float(z_length)))
    return accum_score


studyname="petmr"
path_inference = f"H:/Random/MICCAI_2024/{studyname}/Inference/"
path_analysis = f"H:/Random/MICCAI_2024/{studyname}/Analysis/"
#model_names=['GT','input','BM3D','unet','unet_noMR','transformer','transformer_noMR','diffusion_004096_100','diffusion_002048_100','diffusion_noMR_002048_100']
model_names=['GT','input','dncnn_noMR_002048']

dose_names=['img4ds','img6ds','img8ds','img10ds']

gt_path = os.path.join(path_inference, model_names[0])


filenames=os.listdir(gt_path)

# for each dose
for dose_name in dose_names:

    # for test examples
    for filename in filenames:

        # get input of each dose
        input_path = os.path.join(path_inference, model_names[1])
        input_dose_path = os.path.join(input_path, dose_name)
        input = get_npy(os.path.join(input_dose_path, filename))


        for model_idx in range(2,len(model_names)):
            model_name=model_names[model_idx]
            #goto model
            trg_path = os.path.join(path_inference, model_name)
            #get output of each dose
            trg_dose_path = os.path.join(trg_path, dose_name)
            #get specific example
            trg = get_npy(os.path.join(trg_dose_path, filename))
            trg=input-trg

            recon_path = os.path.join(trg_dose_path, "re_"+filename)

            new_image = nibabel.Nifti1Image(trg, affine=np.eye(4))
            nibabel.save(new_image, recon_path)
