# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from training.pos_embedding import Pos_Embedding
import nibabel

from training.networks import FBPCONVNet

import matplotlib.pyplot as plt


#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

local_save=False

if local_save:
    #----------------------------------------------------------------------------
    rootpath = "/external/syhome"
    #rootpath = "H:"
    itersnames = "002048"
    steps=100

    network_dir=f"{rootpath}/Random/MICCAI_2024/petmr/Models/transformer/network-snapshot-{itersnames}.pkl"

    outroot_gt = f"{rootpath}/Random/MICCAI_2024/petmr/Inference/GT/"
    outroot_input = f"{rootpath}/Random/MICCAI_2024/petmr/Inference/input/"
    outroot_infer = f"{rootpath}/Random/MICCAI_2024/petmr/Inference/transformer/"

    if not os.path.exists(outroot_gt):
        os.makedirs(outroot_gt)
    if not os.path.exists(outroot_input):
        os.makedirs(outroot_input)
    if not os.path.exists(outroot_infer):
        os.makedirs(outroot_infer)

    data_dir = f"{rootpath}/2_Datasets/12_PET_MR7T/dataset_paired/testing/"
else:
    # ----------------------------------------------------------------------------
    rootpath = "/external"
    # rootpath = "H:"
    itersnames = "010000"
    keyword="simpleunet"
    steps = 100

    network_dir = f"{rootpath}/Random/MICCAI_2024/petmr/Models/{keyword}/network-snapshot-{itersnames}.pth"

    outroot_gt = f"{rootpath}/Random/MICCAI_2024/petmr/Inference/GT/"
    outroot_input = f"{rootpath}/Random/MICCAI_2024/petmr/Inference/input/"
    outroot_infer = f"{rootpath}/Random/MICCAI_2024/petmr/Inference/{keyword}_{itersnames}/"

    if not os.path.exists(outroot_gt):
        os.makedirs(outroot_gt)
    if not os.path.exists(outroot_input):
        os.makedirs(outroot_input)
    if not os.path.exists(outroot_infer):
        os.makedirs(outroot_infer)

    data_dir = f"{rootpath}/2_Datasets/12_PET_MR7T/dataset_paired/testing/"

averaging = 1
@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, default = network_dir, required=True)
@click.option('--resolution',              help='Sample resolution', metavar='INT',                                 type=int, default=512)
@click.option('--embed_fq',                help='Positional embedding frequency', metavar='INT',                    type=int, default=0)
@click.option('--mask_pos',                help='Mask out pos channels', metavar='BOOL',                            type=bool, default=False, show_default=True)
@click.option('--on_latents',              help='Generate with latent vae', metavar='BOOL',                            type=bool, default=False, show_default=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str,default = "./results/", required=True)

# patch options
@click.option('--x_start',                 help='Sample resolution', metavar='INT',                                 type=int, default=0)
@click.option('--y_start',                 help='Sample resolution', metavar='INT',                                 type=int, default=0)
@click.option('--image_size',                help='Sample resolution', metavar='INT',                                 type=int, default=None)

@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='1', show_default=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=1, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=steps, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

def main(network_pkl, resolution, on_latents, embed_fq, mask_pos, x_start, y_start, image_size, outdir, subdirs,
         seeds, class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):
    generator = FBPCONVNet().to(device)
    generator.load_state_dict(torch.load(network_dir, map_location=device))


    c = dnnlib.EasyDict()

    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.CustomDataset_PET_Denoise_test_2D', path=data_dir)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2)


    dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=1))

    for batch_seeds in range(len(dataset_iterator)):
        # for batch_seeds in range(1):
        # torch.distributed.barrier()
        dist.print0(f"{batch_seeds} image")
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

        less = less.to(device)
        mr = mr.to(device)

        less = less.to(torch.float32)
        mr = mr.to(torch.float32)
        image_channel = 1

        out_npy=np.zeros((less.shape[-3],less.shape[-2],less.shape[-1]))

        for z_index in range (less.shape[-1]):
            in_src=less[:,:,:,:,z_index]
            in_mr=mr[:,:,:,:,z_index]
            imgs_in= torch.cat([in_src,in_mr],dim=1)


            with torch.no_grad():
                output = generator(imgs_in)
                output = output + in_src
                output = torch.squeeze(output)

                # Save images.
                output = output.cpu().numpy()

                output = (maxs[0][0,z_index]-mins[0][0, z_index]) * output +mins[0][0, z_index]

                # in_src = torch.squeeze(in_src)

                # in_src = in_src.cpu().numpy()
                out_npy[:, :, z_index] = output

        name=names[0][0]
        dose=doses[0][0]




        if not os.path.exists(outroot_infer + dose):
            os.makedirs(outroot_infer + dose)
        recon_path = os.path.join(outroot_infer+ dose, name+ ".nii.gz")


        new_image = nibabel.Nifti1Image(out_npy, affine=np.eye(4))
        nibabel.save(new_image, recon_path)


        full = torch.squeeze(full)

        recon_path = os.path.join(outroot_gt, name+ ".nii.gz")
        if not os.path.exists(recon_path):
            # Save images.
            full = full.numpy()
            new_image = nibabel.Nifti1Image(full, affine=np.eye(4))
            nibabel.save(new_image, recon_path)

        # Save images.
        if not os.path.exists(outroot_input+dose):
            os.makedirs(outroot_input+dose)
        recon_path = os.path.join(outroot_input+dose, name + ".nii.gz")
        if not os.path.exists(recon_path):
            o_less_dose = torch.squeeze(o_less_dose)

            o_less_dose = o_less_dose.cpu().numpy()
            new_image = nibabel.Nifti1Image(o_less_dose, affine=np.eye(4))
            nibabel.save(new_image, recon_path)



        # Done.
        dist.print0('Done.')


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
