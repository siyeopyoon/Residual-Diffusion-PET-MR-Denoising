import nibabel
import os
import multiprocessing

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import torch


class CustomDataset_obl():
    def __init__(self,
                 path,  # Path to directory or zip.
                 cache=True,  # Cache images in CPU memory?
                 patchsize=64,
                 repeat_data=5,
                 angle_half_amp=5.0,
                 ):
        filenames = os.listdir(path)
        self.volume_filenames = [os.path.join(path, x) for x in filenames]
        # self.volume_filenames=self.volume_filenames[0:3]
        self.repeat_data = repeat_data

        self.dataset = []
        self.patchsize = patchsize
        # self.use_normalizer=use_normalizer
        self.cache = cache
        if self.cache:

            for file_idx in range(len(self.volume_filenames)):
                print(f"{file_idx + 1}/{len(self.volume_filenames)} -th pair reading ")

                niftifile=nibabel.load(self.volume_filenames[file_idx])
                volume = niftifile.get_fdata()
                volume = volume.astype(np.float32) / 4096.0  # HU normalization
                volume, _min, _max = normalizer(volume)
                angle_half_amp=0.0
                volume = volume.astype(np.float32)
                views = [{"viewname": "APview", "angle": [-angle_half_amp, angle_half_amp]},
                         {"viewname": "OBLview", "angle": [45 - angle_half_amp, 45 + angle_half_amp]},
                         {"viewname": "LATview", "angle": [90 - angle_half_amp, 90 + angle_half_amp]},
                         ]
                volume=volume.transpose((2,0,1))
                volume_rot = torch.tensor(volume, dtype=torch.float32).unsqueeze(0)
                volume_shape = volume.shape

                for repeat_id in range(repeat_data):
                    print(f"\t {repeat_id + 1}/{repeat_data} -th sub samples with {angle_half_amp} degree perturb")
                    projections = []
                    for view in views:
                        # Get shape of downsized volume

                        # give enough space for rotation
                        zpad = torch.zeros((1, volume_shape[0] , volume_shape[1]* 2, volume_shape[2] * 2))
                        wing_l = volume_shape[1] // 2
                        wing_d = volume_shape[2] // 2
                        zpad[:, :,  wing_l:wing_l + volume_shape[1], wing_d:wing_d + volume_shape[2]] = volume_rot[:, :, :, :]

                        # get rotation angle and generate rotation matrix
                        # generate the random rotation angle using the given range, consider the aspect ratio of volume
                        rangle = view["angle"]
                        angle = np.random.uniform(rangle[0], rangle[1])
                        angulo = torch.pi / 180.0 * float(angle)  # 45 degrees in radians

                        rotation_matrix = torch.tensor(
                            [[np.cos(angulo), np.sin(angulo) * float(volume_shape[1]) / float(volume_shape[2]), 0],
                             [-np.sin(angulo) * float(volume_shape[2]) / float(volume_shape[1]), np.cos(angulo), 0]],
                            dtype=torch.float32)

                        rotation_matrix = rotation_matrix.expand(zpad.size()[0], -1, -1)

                        # generate the affine grid for
                        grid = torch.nn.functional.affine_grid(rotation_matrix, zpad.size(), align_corners=False)

                        # generate the rotated volume
                        resampled = torch.nn.functional.grid_sample(zpad, grid, padding_mode='zeros',
                                                                    align_corners=False)



                        resampled = torch.pow(resampled, 2)

                        # projection and Normalization
                        drr_volume_z = torch.mean(resampled, dim=-1)
                        drr_volume_z = (drr_volume_z - torch.min(drr_volume_z)) / (
                                torch.max(drr_volume_z) - torch.min(drr_volume_z))
                        drr_volume_z = torch.clip(drr_volume_z, 0.0, 1.0)

                        drr_volume_z = drr_volume_z.squeeze()
                        wing_l = (drr_volume_z.shape[1] - volume_shape[1]) // 2
                        drr_volume_z = drr_volume_z[:,wing_l:wing_l + volume_shape[1]]

                        # make "OVERLY" back projection to 3D
                        zpad_radiography = torch.zeros((volume_shape[0], drr_volume_z.shape[1] * 2))
                        zpad_radiography[:,wing_l:wing_l + volume_shape[1]] = drr_volume_z

                        zpad_radiography = zpad_radiography.unsqueeze(-1)
                        zpad_radiography=zpad_radiography.expand(volume_shape[0],
                                                                 drr_volume_z.shape[1] * 2,
                                                                 drr_volume_z.shape[1] * 2)
                        zpad_radiography=zpad_radiography.unsqueeze(0)

                        # inversly rotate to spatial alignment of projections,
                        angulo = torch.pi / 180.0 * float(-angle)  # 45 degrees in radians
                        rotation_matrix = torch.tensor([[np.cos(angulo), np.sin(angulo), 0],
                                                        [-np.sin(angulo), np.cos(angulo), 0]], dtype=torch.float32)

                        rotation_matrix = rotation_matrix.expand(zpad_radiography.size()[0], -1, -1)

                        grid = torch.nn.functional.affine_grid(rotation_matrix, zpad_radiography.size(),
                                                               align_corners=False)


                        zpad_radiography = torch.nn.functional.grid_sample(zpad_radiography, grid, padding_mode='zeros',
                                                                           align_corners=False)

                        zpad_radiography = zpad_radiography.squeeze()
                        zpad_radiography = zpad_radiography.numpy()


                        # make as the target shape.
                        wing_l = (zpad_radiography.shape[1] - volume_shape[1]) // 2
                        wing_d = (zpad_radiography.shape[2] - volume_shape[2]) // 2
                        zpad_radiography = zpad_radiography[:, wing_l:wing_l + volume_shape[1], wing_d:wing_d + volume_shape[2]]
                        zpad_radiography = zpad_radiography.astype(np.float32)

                        volume = volume.transpose((1, 2, 0))
                        zpad_radiography = zpad_radiography.transpose((1, 2, 0))

                        print(
                            f"volume shape :{volume.shape} and dtype: {volume.dtype}, zpad_radiography {zpad_radiography.shape} dtype: {zpad_radiography.dtype}")
                        #
                        projections.append(zpad_radiography)

                    self.dataset.append({"volume": volume, "APview": projections[0], "OBLview": projections[1],
                                         "LATview": projections[2]})

def normalizer( data):
    minval = np.min(data)
    maxval = np.max(data)
    rescaled = (data - minval) / (maxval - minval)
    return rescaled, minval, maxval
if __name__ == '__main__':
    path="H:/2_Datasets/3_WristCTs/FractureCT/"
    dataset = CustomDataset_obl (path)
