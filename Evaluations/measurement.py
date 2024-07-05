import os
import numpy as np
import nibabel
import skimage.metrics as metric
from skimage.filters import threshold_multiotsu
from mahotas.features import haralick
import torch

import torch.nn as nn
import torch
from torchvision.models import vgg19
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg19_model = vgg19(weights=True)
        self.vgg19_54 = nn.Sequential(*list(vgg19_model.features.children())[:35])

    def forward(self, img):

        img = torch.cat([img,img,img],dim=1)

        return self.vgg19_54(img)

FeatureNet= FeatureExtractor()

def get_perception (src,trg, min_val, max_val):
    psrc=min_max_norm(src,min_val,max_val)
    ptrg=min_max_norm(trg,min_val,max_val)

    src_slice = torch.tensor(psrc).unsqueeze(0).unsqueeze(0)
    trg_slice = torch.tensor(ptrg).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        a = FeatureNet(src_slice).squeeze().numpy()
        b = FeatureNet(trg_slice).squeeze().numpy()



    return np.mean((a-b)**2)

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
    return np.sqrt(((img1_feature-img2_feature)**2)/(img2_feature**2)) #np.linalg.norm()


def get_haralic (src,trg,min_val,max_val):
    psrc=min_max_norm(src,min_val,max_val)
    ptrg=min_max_norm(trg,min_val,max_val)
    z_length=psrc.shape[-1]

    accum_score = get_haralic_slice(psrc, ptrg)

    accum_score=np.sum(accum_score)
    return accum_score


studyname="petmr"
path_inference = f"H:/Random/MICCAI_2024/{studyname}/Inference/"
path_analysis = f"H:/Random/MICCAI_2024/{studyname}/Analysis/"
#model_names=['GT','input','BM3D','unet','unet_noMR','transformer','transformer_noMR','diffusion_004096_100','diffusion_002048_100','diffusion_noMR_002048_100']
#model_names=['GT','input','denoise_tv_chambolle','simpleunet_010000','transformer_002048','diffusion_004096_100','diffusion_noMR_002048_100']
model_names=['GT','input','denoise_tv_chambolle','simpleunet_010000','transformer_002048','diffusion_004096_100','diffusion_noMR_002048_100']

dose_names=['img10ds','img8ds','img6ds','img4ds',]

gt_path = os.path.join(path_inference, model_names[0])


filenames=os.listdir(gt_path)

# for each dose
for dose_name in dose_names:


    # validated values of each model.
    mse_dose = []*(len(model_names))
    mae_dose = []*(len(model_names))
    PSNR_dose = []*(len(model_names))
    ssim_dose = []*(len(model_names))
    haralic_dose = []*(len(model_names))
    # for test examples

    item_count=0


    file_Analysis=os.path.join(path_analysis,f"{dose_name}.txt")
    logger=open(file_Analysis,"w")

    for filename in filenames:

        # get gt and mask
        GT = get_npy(os.path.join(gt_path, filename))
        mask = GT > 0.0
        GT = GT

        # get input of each dose
        input_path = os.path.join(path_inference, model_names[1])
        input_dose_path = os.path.join(input_path, dose_name)
        input = get_npy(os.path.join(input_dose_path, filename))
        input = input

        # to be normalized
        #min_val = np.min(input)
        min_val = 0.0
        max_val = np.max(input)




        for model_idx in range(1,len(model_names)):
            model_name=model_names[model_idx]
            #goto model
            trg_path = os.path.join(path_inference, model_name)
            #get output of each dose
            trg_dose_path = os.path.join(trg_path, dose_name)
            #get specific example
            trg = get_npy(os.path.join(trg_dose_path, filename))
            trg = trg

            for z_idx in range (10,GT.shape[-1]-10):
                GT_slice= GT[:,:,z_idx]
                trg_slice = trg[:, :, z_idx]


                mse = get_MSE(GT_slice, trg_slice)
                mae = get_MAE(GT_slice, trg_slice)
                PSNR = get_PSNR(GT_slice, trg_slice, min_val, max_val)
                ssim = get_SSIM(GT_slice, trg_slice, min_val, max_val)
                haralic = get_haralic(GT_slice, trg_slice, min_val, max_val)
                perception=get_perception(GT_slice, trg_slice, min_val, max_val)
                print (dose_name,filename, model_name,z_idx,mse,mae,PSNR,ssim,haralic,perception)
                logger.write(f"{dose_name},{model_name},{z_idx},{mse},{mae},{PSNR},{ssim},{haralic},{perception}\n")

    logger.close()











