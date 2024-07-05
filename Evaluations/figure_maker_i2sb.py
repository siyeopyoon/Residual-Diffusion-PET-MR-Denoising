import os
import numpy as np
import nibabel
import skimage.metrics as metric
from skimage.filters import threshold_multiotsu
from mahotas.features import haralick
import torch
import matplotlib.pyplot as plt
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

def zpad_data(data_volume,fixer):


    xdif = 0
    if data_volume.shape[0] - (data_volume.shape[0] // fixer) * fixer > 0:
        xdif = ((data_volume.shape[0] // fixer) + 1) * fixer - data_volume.shape[0]

    ydif = 0
    if data_volume.shape[1] - (data_volume.shape[1] // fixer) * fixer > 0:
        ydif = ((data_volume.shape[1] // fixer) + 1) * fixer - data_volume.shape[1]

    zdif = 0
    if data_volume.shape[2] - (data_volume.shape[2] // fixer) * fixer > 0:
        zdif = ((data_volume.shape[2] // fixer) + 1) * fixer - data_volume.shape[2]

    volume_extended = np.zeros((xdif + data_volume.shape[0], ydif + data_volume.shape[1], zdif + data_volume.shape[2]))
    volume_extended[xdif // 2:xdif // 2 + data_volume.shape[0], ydif // 2:ydif // 2 + data_volume.shape[1], zdif // 2:zdif // 2 + data_volume.shape[2]] = data_volume
    return volume_extended
studyname="petmr"
path_inference = f"H:/Random/MICCAI_2024/{studyname}/Inference/"
path_analysis = f"H:/Random/MICCAI_2024/{studyname}/Analysis_i2sb_100/"
path_figure = f"H:/Random/MICCAI_2024/{studyname}/Figure_i2sb_100/"

#model_names=['GT','input','BM3D','unet','unet_noMR','transformer','transformer_noMR','diffusion_004096_100','diffusion_002048_100','diffusion_noMR_002048_100']
model_names=['GT','input','denoise_tv_chambolle','simpleunet_010000','transformer_002048','i2sb_100']

dose_names=['img10ds']

gt_path = os.path.join(path_inference, model_names[0])


filenames=os.listdir(gt_path)


mr_path = os.path.join(path_inference, 'mr')

# for each dose
for dose_name in dose_names:


    # validated values of each model.
    mse_dose = np.zeros(len(model_names))
    mae_dose = np.zeros(len(model_names))
    PSNR_dose = np.zeros(len(model_names))
    ssim_dose = np.zeros(len(model_names))
    haralic_dose = np.zeros(len(model_names))
    # for test examples



    for filename in filenames:
        method_volumes = []

        MR = get_npy(os.path.join(mr_path, filename))
        MR=zpad_data(MR,8)
        mask = MR > 0.0
        MR = MR * mask

        MR=(MR-np.min(MR))/(np.percentile(MR,99)-np.min(MR))
        MR=np.clip(MR,0.0,1.0)

        # get gt and mask
        GT = get_npy(os.path.join(gt_path, filename))

        GT = GT*mask
        # to be normalized
        min_val = np.min(GT)
        max_val = np.max(GT)
        #max_val = 1.0

        #GT=np.clip(GT, min_val,max_val)
        method_volumes.append(GT)
        # get input of each dose
        input_path = os.path.join(path_inference, model_names[1])
        input_dose_path = os.path.join(input_path, dose_name)
        input = get_npy(os.path.join(input_dose_path, filename))
        input = input*mask


        #input=np.clip(input, min_val,max_val)


        method_volumes.append(input)


        for model_idx in range(2,len(model_names)):
            model_name=model_names[model_idx]
            #goto model
            trg_path = os.path.join(path_inference, model_name)
            #get output of each dose
            trg_dose_path = os.path.join(trg_path, dose_name)
            #get specific example
            trg = get_npy(os.path.join(trg_dose_path, filename))
            trg = trg*mask

            #trg = np.clip(trg, min_val, max_val)
            method_volumes.append(trg)


        error_volumes=[]
        for method_volume in method_volumes:
            error_volumes.append(method_volume-GT)


        for idx in range(GT.shape[-1]):

            slice = []
            for method_volume in method_volumes:
                img=np.rot90(method_volume[:,:,idx])
                slice.append(img)

            slice = np.concatenate(slice, axis=1)

            color_bar=np.zeros((slice.shape[0],10))

            for i_x in range (slice.shape[0]):
                color_bar[i_x]= +1.-1.*(i_x)/float(slice.shape[0]-1)
            slice=np.concatenate([slice,color_bar],axis=1)
            slice=np.clip(slice,0.0,1.)


            slice_err = []
            for error_volume in error_volumes:
                img=np.rot90(error_volume[:,:,idx])
                slice_err.append(img)

            slice_err= np.concatenate(slice_err,axis=1)

            color_bar=np.zeros((slice_err.shape[0],10))


            for i_x in range (slice_err.shape[0]):
                color_bar[i_x]= +1.-2.0*(i_x)/float(slice_err.shape[0]-1)
            slice_err=np.concatenate([slice_err,color_bar],axis=1)
            slice_err=np.clip(slice_err,-1.,1.)


            if not os.path.exists(f"{path_figure}/{dose_name}/Z-slice"):
                os.makedirs(f"{path_figure}/{dose_name}/Z-slice")
            plt.imsave(f"{path_figure}/{dose_name}/Z-slice/{filename[:-7]}_{idx}.tiff", slice, cmap='gray_r')

            plt.imsave(f"{path_figure}/{dose_name}/Z-slice/err_{filename[:-7]}_{idx}.tiff", slice_err, cmap='bwr')


            mrslice = MR[:, :, idx]
            mrslice = np.rot90(mrslice)
            plt.imsave(f"{path_figure}/{dose_name}/Z-slice/mr_{filename[:-7]}_{idx}.tiff", mrslice, cmap='gray')

        for idx in range(GT.shape[-2]):

            slice = []
            for method_volume in method_volumes:
                img = np.rot90(method_volume[:, idx,:])
                slice.append(img)

            slice = np.concatenate(slice, axis=1)

            color_bar=np.zeros((slice.shape[0],10))

            for i_x in range (slice.shape[0]):
                color_bar[i_x]= +1.-1.*(i_x)/float(slice.shape[0]-1)
            slice=np.concatenate([slice,color_bar],axis=1)
            slice=np.clip(slice,0.0,1.)

            slice_err = []
            for error_volume in error_volumes:
                img = np.rot90(error_volume[:, idx, :])
                slice_err.append(img)

            slice_err = np.concatenate(slice_err, axis=1)

            color_bar=np.zeros((slice_err.shape[0],10))



            for i_x in range (slice_err.shape[0]):
                color_bar[i_x]= +1.-2.0*(i_x)/float(slice_err.shape[0]-1)
            slice_err=np.concatenate([slice_err,color_bar],axis=1)
            slice_err=np.clip(slice_err,-1.,1.)

            if not os.path.exists(f"{path_figure}/{dose_name}/Y-slice"):
                os.makedirs(f"{path_figure}/{dose_name}/Y-slice")
            plt.imsave(f"{path_figure}/{dose_name}/Y-slice/{filename[:-7]}_{idx}.tiff", slice, cmap='gray_r')

            plt.imsave(f"{path_figure}/{dose_name}/Y-slice/err_{filename[:-7]}_{idx}.tiff", slice_err, cmap='bwr')

            mrslice = MR[:, idx,:]
            mrslice = np.rot90(mrslice)
            plt.imsave(f"{path_figure}/{dose_name}/Y-slice/mr_{filename[:-7]}_{idx}.tiff", mrslice, cmap='gray')


        for idx in range(GT.shape[-3]):

            slice = []
            for method_volume in method_volumes:
                img = np.rot90(method_volume[idx,:, :])
                slice.append(img)

            slice = np.concatenate(slice, axis=1)

            color_bar=np.zeros((slice.shape[0],10))

            for i_x in range (slice.shape[0]):
                color_bar[i_x]= +1.-1.*(i_x)/float(slice.shape[0]-1)
            slice=np.concatenate([slice,color_bar],axis=1)
            slice=np.clip(slice,0.0,1.)



            slice_err = []
            for error_volume in error_volumes:
                img = np.rot90(error_volume[idx,:,  :])
                slice_err.append(img)

            slice_err = np.concatenate(slice_err, axis=1)

            color_bar=np.zeros((slice_err.shape[0],10))



            for i_x in range (slice_err.shape[0]):
                color_bar[i_x]= +1.-2.0*(i_x)/float(slice_err.shape[0]-1)
            slice_err=np.concatenate([slice_err,color_bar],axis=1)
            slice_err=np.clip(slice_err,-1.,1.)


            if not os.path.exists(f"{path_figure}/{dose_name}/X-slice"):
                os.makedirs(f"{path_figure}/{dose_name}/X-slice")
            plt.imsave(f"{path_figure}/{dose_name}/X-slice/{filename[:-7]}_{idx}.tiff", slice, cmap='gray_r')
            plt.imsave(f"{path_figure}/{dose_name}/X-slice/err_{filename[:-7]}_{idx}.tiff", slice_err, cmap='bwr')

            mrslice = MR[idx,:,  :]
            mrslice = np.rot90(mrslice)
            plt.imsave(f"{path_figure}/{dose_name}/X-slice/mr_{filename[:-7]}_{idx}.tiff", mrslice, cmap='gray')


