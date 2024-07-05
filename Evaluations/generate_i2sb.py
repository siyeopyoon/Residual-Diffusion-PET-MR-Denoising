import torch
from denoising_diffusion_pytorch.schrodinger_bridge import Distinguish_Unet, GaussianDiffusion, Trainer, HL_test_Dataset,CustomDataset_PET_Denoise_test_2D
import gecatsim as xc
from multiprocessing import cpu_count
from torch.utils.data import DataLoader
import numpy as np
from matplotlib import pyplot as plt
import math,nibabel
import os

if __name__ == '__main__':
    model = Distinguish_Unet(
        dim=64,
        cond_dim = 32,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        cond_channels = 2,
        flash_attn=False
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=160,
        timesteps=1000,  # number of steps
        sampling_timesteps=1000,
        beta_max=0.3
        # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    )

    external = "/external"
    data_dir = f"{external}/2_Datasets/12_PET_MR7T/dataset_paired/testing/"

    model_folder = f"{external}/0_TrainingOutputs/4_ScBridge/I2SB/samples_more/"

    outroot_gt = f"{external}/Random/MICCAI_2024/petmr/Inference/GT/"
    outroot_input = f"{external}/Random/MICCAI_2024/petmr/Inference/input/"

    outroot_infer = f"{external}/Random/MICCAI_2024/petmr/Inference/"
    trainer = Trainer(
        diffusion,
        data_dir,
        train_batch_size=16,
        train_lr=8e-5,
        train_num_steps=300000,  # total training steps
        gradient_accumulate_every=1,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        calculate_fid=False,  # whether to calculate fid during training
        results_folder = outroot_infer,
        original_image_size = 160,
        num_samples = 1,
        save_and_sample_every = 10000,
        residual = False,
        position_encoding = True
    )
    print(torch.cuda.memory_allocated())
    trainer.load_path(model_folder+"model-100.pt")



    step_list = [10,50,100]

    for steps in reversed(step_list):
        test_ds = CustomDataset_PET_Denoise_test_2D(data_dir)
        dataset_iterator = iter(torch.utils.data.DataLoader(dataset=test_ds, batch_size=1))

        for batch_seeds in range(len(dataset_iterator)):

            outroot_infer = f"{external}/Random/MICCAI_2024/petmr/Inference/i2sb_{steps}/"
            os.makedirs(outroot_infer, exist_ok=True)

            # for batch_seeds in range(1):
            # torch.distributed.barrier()
            print(f"{batch_seeds} image")
            batch_size = 1
            # batch_mul = batch_mul_dict[patch_size]
            mr, full, less, mins, maxs, names, doses, o_less_dose = [], [], [], [], [], [], [], []
            for _ in range(batch_size):  # batch size per gpu
                mr_, full_, less_, min_, max_, name_, dose_, o_less_dose_ = next(dataset_iterator)
                mr.append(mr_)
                full.append(full_)
                less.append(less_)
                mins.append(min_)
                maxs.append(max_)

                names.append(name_)
                doses.append(dose_)
                o_less_dose.append(o_less_dose_)

            mr = torch.cat(mr, dim=0)
            full = torch.cat(full, dim=0)
            less = torch.cat(less, dim=0)
            o_less_dose = torch.cat(o_less_dose, dim=0)
            del mr_, full_, less_, min_, max_, name_, dose_, o_less_dose_

            less = less.to(torch.float32)
            mr = mr.to(torch.float32)



            incat=torch.concatenate([mr,less],dim=1)
            image_channel = 1

            out_npy=np.zeros((less.shape[-3],less.shape[-2],less.shape[-1]))

            incat = incat.to("cuda")


            for z_index in range (less.shape[-1]):
                in_src=incat[:,:,:,:,z_index]
                print (z_index," of ",less.shape[-1])
                with torch.no_grad():
                    output=diffusion.sample(in_src, image_size=160, batch_size=1, ddim_sample=True, ita=1, sampling_timesteps=steps)

                    output = torch.squeeze(output)

                    # Save images.
                    output = output.cpu().numpy()

                    output = (maxs[0][0,z_index]-mins[0][0, z_index]) * output +mins[0][0, z_index]

                    # in_src = torch.squeeze(in_src)

                    # in_src = in_src.cpu().numpy()
                    out_npy[:, :, z_index] = output

            name = names[0][0]
            dose = doses[0][0]

            if not os.path.exists(outroot_infer + dose):
                os.makedirs(outroot_infer + dose)
            recon_path = os.path.join(outroot_infer + dose, name + ".nii.gz")

            new_image = nibabel.Nifti1Image(out_npy, affine=np.eye(4))
            nibabel.save(new_image, recon_path)

            full = torch.squeeze(full)

            recon_path = os.path.join(outroot_gt, name + ".nii.gz")
            if not os.path.exists(recon_path):
                # Save images.
                full = full.numpy()
                new_image = nibabel.Nifti1Image(full, affine=np.eye(4))
                nibabel.save(new_image, recon_path)

            # Save images.
            if not os.path.exists(outroot_input + dose):
                os.makedirs(outroot_input + dose)
            recon_path = os.path.join(outroot_input + dose, name + ".nii.gz")
            if not os.path.exists(recon_path):
                o_less_dose = torch.squeeze(o_less_dose)

                o_less_dose = o_less_dose.cpu().numpy()
                new_image = nibabel.Nifti1Image(o_less_dose, affine=np.eye(4))
                nibabel.save(new_image, recon_path)

            # Done.
            print('Done.')