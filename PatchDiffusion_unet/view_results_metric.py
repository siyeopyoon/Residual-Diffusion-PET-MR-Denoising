import torch
import numpy as np
import os
import nibabel
import matplotlib.pyplot as plt
import glob
import scipy.ndimage as sci
import skimage.metrics as metric


def main ():
    itersname = "001638"


    doselist=['img4ds','img6ds','img8ds','img10ds']


    directory = f"H:/Random/12_PET_MR7T/{itersname}/"

    generated_suff=f"generated_network{itersname}"


    plist=os.listdir(directory)
    p_fulls=[]
    for p_item in plist:
        if "full_original.npy" in p_item:
            p_fulls.append(p_item)



    for p_full in p_fulls:
        for lowdose in doselist:
            path_full= directory+p_full

            pid_spt=p_full.split('_')

            path_less = directory+pid_spt[0]+"_"+pid_spt[1]+"_"+lowdose+"_less.npy"

            path_gen =directory+ pid_spt[0]+"_"+pid_spt[1]+"_"+lowdose+"_"+generated_suff+".npy"

            full= np.load(path_full)
            mask = np.where(full > 0, 1, 0)
            less= np.load(path_less)
            gen= np.load(path_gen)

            gen_masked=mask*gen

            max= np.max([np.max(full),np.max(less),np.max(gen_masked)])
            #mse= metric.structural_similarity(full, less,data_range=max)
            #mse_gen= metric.structural_similarity(full, gen_masked,data_range=max)

            MAE=np.average(np.sqrt(np.power(full-less,2)))
            MAE_gen = np.average(np.sqrt(np.power(full- gen_masked,2)))

            print (pid_spt[0]+"_"+lowdose, MAE,  MAE_gen)




if __name__ == "__main__":
    main()

