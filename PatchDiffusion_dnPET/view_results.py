import torch
import numpy as np
import os
import nibabel
import matplotlib.pyplot as plt
import glob
import scipy.ndimage as sci
def main ():
    itersnames = ["001638-c"]
    directory = f"H:/Random/12_PET_MR7T/{itersnames[0]}/"
    npy_files = glob.glob(os.path.join(directory, '**/*.npy'), recursive=True)


    for npy_file in reversed(npy_files):
        datagen = np.load(npy_file)
        #datagen=datagen-np.min(datagen)
        datagen[datagen<0.0]=0.0

        basename = os.path.basename(npy_file)
        new_image = nibabel.Nifti1Image(datagen, affine=np.eye(4))
        nibabel.save(new_image, f"{directory}/{basename[:-4]}.nii")
        datagencut = datagen[:, datagen.shape[1] // 2, :]
        plt.imsave(f"{directory}/{basename[:-4]}.tiff", datagencut, cmap='gray')

        plt.imsave(f"{directory}/{basename[:-4]}_LAT.tiff", np.average(datagen**2,axis=1), cmap='gray')

        plt.imsave(f"{directory}/{basename[:-4]}_PA.tiff", np.average(datagen**2,axis=0), cmap='gray')



if __name__ == "__main__":
    main()

