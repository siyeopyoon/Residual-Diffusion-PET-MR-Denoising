import math
import copy
from pathlib import Path
from random import random
from random import randint
import scipy.stats as st

from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
import gecatsim as xc

import torch
from torch import nn, einsum
from torch.cuda.amp import autocast
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from denoising_diffusion_pytorch.attend import Attend
from denoising_diffusion_pytorch.fid_evaluation import FIDEvaluation

from denoising_diffusion_pytorch.version import __version__
import numpy as np
import os,nibabel


# constants

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def cast_tuple(t, length = 1):
    if isinstance(t, tuple):
        return t
    return ((t,) * length)

def divisible_by(numer, denom):
    return (numer % denom) == 0

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image

# normalization functions

def normalize_to_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5

# small helper modules

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * self.g * (x.shape[1] ** 0.5)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert divisible_by(dim, 2)
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            RMSNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -1), ((mk, k), (mv, v)))

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        dim_head = 32,
        num_mem_kv = 4,
        flash = False
    ):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.attend = Attend(flash = flash)

        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h = self.heads), qkv)

        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b = b), self.mem_kv)
        k, v = map(partial(torch.cat, dim = -2), ((mk, k), (mv, v)))

        out = self.attend(q, k, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(out)

# model
class Distinguish_Unet(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        cond_channels = 1,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (False, False, False, True),
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.cond_channels = cond_channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim-cond_dim, 7, padding = 3)
        self.init_cond_conv = nn.Conv2d(cond_channels, cond_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, cond, time, x_self_cond = None):
        
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        
        #x = torch.cat((x, cond), dim=1)
        x = self.init_conv(x)
        cond = self.init_cond_conv(cond)
        x = torch.cat((x, cond), dim=1)

        r = x.clone()
        t = self.time_mlp(time)

        h = []
        
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        
        return self.final_conv(x)



class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults = (1, 2, 4, 8),
        channels = 3,
        cond_channels = 1,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        sinusoidal_pos_emb_theta = 10000,
        attn_dim_head = 32,
        attn_heads = 4,
        full_attn = (False, False, False, True),
        flash_attn = False
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.cond_channels = cond_channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels+cond_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta = sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # attention

        num_stages = len(dim_mults)
        full_attn  = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash = flash_attn)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                attn_klass(dim_in, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = FullAttention(mid_dim, heads = attn_heads[-1], dim_head = attn_dim_head[-1])
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))):
            is_last = ind == (len(in_out) - 1)

            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                attn_klass(dim_out, dim_head = layer_attn_dim_head, heads = layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, cond, time, x_self_cond = None):
        
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), f'your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)
        x = torch.cat((x, cond), dim=1)
        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        h = []
        
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)
        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        
        return self.final_conv(x)

# gaussian diffusion trainer class

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def make_beta_schedule(n_timestep=1000, linear_start=1e-4, linear_end=2e-2):
    # return np.linspace(linear_start, linear_end, n_timestep)
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float32) ** 2
    )
    return betas.numpy()
def compute_gaussian_product_coef(sigma1, sigma2):
    """ Given p1 = N(x_t|x_0, sigma_1**2) and p2 = N(x_t|x_1, sigma_2**2)
        return p1 * p2 = N(x_t| coef1 * x0 + coef2 * x1, var) """

    denom = sigma1**2 + sigma2**2
    coef1 = sigma2**2 / denom
    coef2 = sigma1**2 / denom
    var = (sigma1**2 * sigma2**2) / denom
    return coef1, coef2, var
def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]
    
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps = 10,
        sampling_timesteps = None,
        objective = 'pred_noise',
        ddim_sampling_eta = 0.,
        auto_normalize = True,
        offset_noise_strength = 0.,  # https://www.crosslabs.org/blog/diffusion-with-offset-noise
        beta_max = 0.3
    ):
        super().__init__()
        assert not (type(self) == GaussianDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model

        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.image_size = image_size

        self.objective = objective

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'
        self.beta_max = beta_max
        betas = make_beta_schedule(n_timestep=timesteps, linear_end=beta_max/timesteps)
        betas = np.concatenate([betas[:timesteps//2], np.flip(betas[:timesteps//2])])
        
        std_fwd = np.sqrt(np.cumsum(betas))
        std_bwd = np.sqrt(np.flip(np.cumsum(np.flip(betas))))

        std_bwd[0:len(std_bwd)-1] = std_bwd[1:len(std_bwd)]
        std_bwd[len(std_bwd)-1] = 0

        mu_x0, mu_x1, var = compute_gaussian_product_coef(std_fwd, std_bwd)

        #print(mu_x0)
        #print(mu_x1)
        std_sb = np.sqrt(var)
       
        # tensorize everything
        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.betas = to_torch(betas).to("cuda")
        self.std_fwd = to_torch(std_fwd).to("cuda")
        self.std_bwd = to_torch(std_bwd).to("cuda")
        self.std_sb  = to_torch(std_sb).to("cuda")
        self.mu_x0 = to_torch(mu_x0).to("cuda")
        self.mu_x1 = to_torch(mu_x1).to("cuda")
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32


        self.offset_noise_strength = offset_noise_strength

        # derive loss weight
        # snr - signal noise ratio


        # auto-normalization of data [0, 1] -> [-1, 1] - can turn off by setting it to be False

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    @property
    def device(self):
        return self.betas.device

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def compute_pred_x0(self, step, xt, net_out, clip_denoise=False):
        """ Given network output, recover x0. This should be the inverse of Eq 12 """
        std_fwd = self.get_std_fwd(step, xdim=xt.shape[1:])
        pred_x0 = xt - std_fwd * net_out
        if clip_denoise: pred_x0.clamp_(-1., 1.)
        return pred_x0

    def model_predictions(self, x, cond, t, x_self_cond = None, clip_x_start=False):
        
        model_output = self.model(x, cond, t, x_self_cond)
        x_start = self.compute_pred_x0(t, x, model_output, clip_x_start)

        return x_start

    def p_mean_variance(self, x, cond, t, x_self_cond = None, clip_denoised = True):
        
        preds = self.model_predictions(x, cond, t, x_self_cond)
        
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, cond,  t: int, x_self_cond = None):
        b, *_, device = *x.shape, self.device
        batched_times = torch.full((b,), t, device = device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x = x, cond = cond, t = batched_times, x_self_cond = x_self_cond, clip_denoised = False)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.inference_mode()
    def p_sample_loop(self, cond, shape, return_all_timesteps = False):
        batch, device = shape[0], self.device

        img = cond[:,0:self.channels,:,:]
        if return_all_timesteps:
            imgs = [img]

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc = 'sampling loop time step', total = self.num_timesteps):
            
            time_cond = torch.full((batch,), t, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            x_start = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = False)

            if t == 0:
                img = x_start
                if return_all_timesteps:
                    imgs.append(img)
                continue         

            img = self.p_posterior(t-1, t, img, x_start)

            if return_all_timesteps:
                imgs.append(img)


        ret = img if not return_all_timesteps else torch.stack(imgs, dim = 1)

        ret = self.unnormalize(ret)
        return ret

    def ddim_p_posterior(self, nprev, n, x_n, x_N, x0, ita):
        assert nprev < n
        std_n     = self.std_fwd[n]   #sigma_(n+1)
        std_nprev = self.std_fwd[nprev]   #sigma_(n)
        std_delta = (std_n**2 - std_nprev**2).sqrt() #alpha_n
        std_bar_nprev = self.std_bwd[nprev] #sigma_bar(n)
        std_bar_n = self.std_bwd[n] #sigma_bar(n+1)

        gn = std_nprev*std_delta/(((std_nprev**2)+(std_delta**2)).sqrt())
        gn = ita * gn
        
        gn_max = (((std_nprev**2)*(std_bar_nprev**2))/((std_nprev**2)+(std_bar_nprev**2))).sqrt()
        #gn = min(gn_max, gn)
        
        xt_prev = (std_bar_nprev**2)*x0/((std_bar_nprev**2)+(std_nprev**2))
        xt_prev = xt_prev + (std_nprev**2)*x_N/((std_bar_nprev**2)+(std_nprev**2))

        if gn<gn_max:
            k = (((std_nprev**2)*(std_bar_nprev**2))-((gn**2)*((std_nprev**2)+(std_bar_nprev**2)))).sqrt()/(std_n*std_bar_n)
            xt_prev = xt_prev + k * x_n

            xt_prev = xt_prev - k *(((std_bar_n**2)*x0/((std_bar_n**2)+(std_n**2)))+((std_n**2)*x_N/((std_bar_n**2)+(std_n**2))))
        else:
            gn = gn_max
        xt_prev = xt_prev + gn* torch.randn_like(xt_prev)     
        
        #mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)
        #xt_prev2 = mu_x0 * x0 + mu_xn * x_n
          
        return xt_prev
    def p_posterior(self, nprev, n, x_n, x0):
        """ Sample p(x_{nprev} | x_n, x_0), i.e. eq 4"""

        assert nprev < n
        std_n     = self.std_fwd[n]
        std_nprev = self.std_fwd[nprev]
        std_delta = (std_n**2 - std_nprev**2).sqrt()

        mu_x0, mu_xn, var = compute_gaussian_product_coef(std_nprev, std_delta)

        xt_prev = mu_x0 * x0 + mu_xn * x_n
        #print(ot_ode)
        #if not ot_ode and nprev > 0:
        xt_prev = xt_prev + var.sqrt() * torch.randn_like(xt_prev)

        return xt_prev

    @torch.inference_mode()
    def ddim_sample(self, cond, shape,nfe,ita, return_all_timesteps = False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        sampling_timesteps = nfe
        eta = ita
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = cond[:,0:self.channels,:,:]
        #imgs = [img.detach()]
        imgs = []
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            x_start = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = False)

            if time_next < 0:
                img = x_start
                imgs.append(img.detach())
                continue
            
            if self.std_bwd[time] < 1e-4:
                img = self.p_posterior(time_next, time, img, x_start)
            else:
                img = self.ddim_p_posterior(time_next, time, img, cond[:,0:self.channels,:,:], x_start, eta)

            if return_all_timesteps:
                imgs.append(x_start.detach())

        ret = img if not return_all_timesteps else torch.flip(torch.stack(imgs, dim = 1), dims=(1,))

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def sample(self, cond, image_size=128, batch_size = 16, return_all_timesteps = False, ddim_sample=False, sampling_timesteps=1000, ita=0.0):
        cond = self.normalize(cond)
        channels = self.channels
        if ddim_sample:
            return self.ddim_sample(cond, (batch_size, channels, image_size, image_size), sampling_timesteps,ita, return_all_timesteps = return_all_timesteps)
        #sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        else:
            sample_fn = self.p_sample_loop
            return sample_fn(cond, (batch_size, channels, image_size, image_size), return_all_timesteps = return_all_timesteps)

    def calc_projection(self, Img, SM):
        batch = Img.shape[0]
        channel = Img.shape[1]
        Img = torch.reshape(Img, [batch*channel, -1])
        projection = torch.sparse.mm(SM, Img.T).T
        return torch.reshape(projection, [batch, channel, -1])

    def back_projection(self, SM_T, projection, Img_size):
        batch = projection.shape[0]
        channel = projection.shape[1]
        projection = torch.reshape(projection, [batch*channel, -1])
        Imgs = torch.sparse.mm(SM_T, projection.T).T
        return torch. reshape(Imgs, [batch, channel, Img_size, Img_size])

    def CG(self, A, x0, y, N):
        """
        Solving Ax=y starting from x0
        A is a function to calculate Ax
        x0.shape [batch, channel, img_size, img_size]
        y.shape [batch, channel, img_size, img_size]
        N: iteration times
        Ax shape [batch, channel, img_size, img_size]
        """
        r = y - A(x0)
        p = r
        x = x0
        for k in range(N):
            Ap = A(p) #[batch, channel, img_size, img_size]
            pTAp = torch.sum(p*Ap, dim=[2,3],keepdim=True)
            rTr = torch.sum(r*r, dim=[2,3],keepdim=True)
            if torch.mean(rTr)<1e-6:
                break
            alpha = rTr/pTAp
            #print(alpha)
            x = x+alpha*p
            r = r-alpha*Ap
            beta = torch.sum(r*r, dim=[2,3],keepdim=True)/rTr
            p=r+beta*p
        return x


    @torch.inference_mode()
    def I4SB_sample(self, cond, SM, SM_T,measurement, ke, ky, CG_iteration=5, image_size=512, batch_size = 16, return_all_timesteps = False, sampling_timesteps=1000):
        cond = self.normalize(cond)
        channels = self.channels
        batch, device, total_timesteps,  objective = batch_size, self.device, self.num_timesteps, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = cond[:,0:self.channels,:,:]
        x_N = cond[:,0:self.channels,:,:]
        #imgs = [img.detach()]
        after_CG_imgs = []
        before_CG_imgs = []
        x_start = None
        ATmeasure = self.back_projection(SM_T, measurement, image_size)
        
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            x_start = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = False)

            if time_next < 0:
                img = x_start
                before_CG_imgs.append(img.detach())
                continue                             
            
            sigma_N = self.std_fwd[total_timesteps - 1]
            sigma_n_bar = self.std_bwd[time]
            sigma_n = self.std_fwd[time]
            if sigma_n_bar < 1e-4:
                x0e = torch.zeros_like(img)
            else:
                x0e = (sigma_N*sigma_N)*img/(sigma_n_bar*sigma_n_bar)-(sigma_n*sigma_n)*x_N/(sigma_n_bar*sigma_n_bar)
            
            if return_all_timesteps:
                before_CG_imgs.append(x_start.detach())
            x0e = (x0e+1)/2
            x_start = (x_start+1)/2
            
            def A(x, SM=SM, img_size=image_size, ke=ke, ky=ky):
                Ax=self.calc_projection(x, SM)
                
                ATAx=self.back_projection(SM_T,Ax,img_size)
                return (1+ke)*x+ky*ATAx
            y = ke*x0e+x_start+ky*ATmeasure
            x_start = self.CG(A,x_start,y,CG_iteration)

            x_start = 2*x_start-1

            img = self.p_posterior(time_next, time, img, x_start)

            if return_all_timesteps:
                after_CG_imgs.append(x_start.detach())
        
        if not return_all_timesteps:
            return self.unnormalize(img)
        else:
            before_CG_imgs = self.unnormalize(torch.flip(torch.stack(before_CG_imgs, dim = 1), dims=(1,)))
            after_CG_imgs = self.unnormalize(torch.flip(torch.stack(after_CG_imgs, dim = 1), dims=(1,)))
            return [before_CG_imgs, after_CG_imgs]

    @torch.inference_mode()
    def I4SB_v2_sample_gN_ky(self, cond, SM, SM_T,measurement, gN, ky, CG_iteration=5, image_size=512, batch_size = 16, return_all_timesteps = False, ddim_sample=False, sampling_timesteps=1000, last_step_consistency=False):
        cond = self.normalize(cond)
        channels = self.channels
        batch, device, total_timesteps,  objective = batch_size, self.device, self.num_timesteps, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = cond[:,0:self.channels,:,:]
        x_N = cond[:,0:self.channels,:,:]
        #imgs = [img.detach()]
        imgs = []
        x_start = None
        ATmeasure = self.back_projection(SM_T, measurement, image_size)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            x_start_prev = x_start
            x_start = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = False)

            if (time_next < 0)&(last_step_consistency==False):
                img = x_start
                imgs.append(img.detach())
                continue                  
            
            
            sigma_N = self.std_fwd[total_timesteps - 1]
            sigma_n_bar = self.std_bwd[time]
            sigma_n = self.std_fwd[time]

            if sigma_n_bar < 1e-4:
                x0e = torch.zeros_like(img)
            else:
                x0e = x_start_prev              
            
            x0e = (x0e+1)/2
            x_start = (x_start+1)/2

            ke = gN*(sigma_n_bar*sigma_n_bar)/(sigma_N*sigma_N)
            
            def A(x, SM=SM, img_size=image_size, ke=ke, ky=ky):
                Ax=self.calc_projection(x, SM)
                
                ATAx=self.back_projection(SM_T,Ax,img_size)
                return (1+ke)*x+ky*ATAx
            y = ke*x0e+x_start+ky*ATmeasure
            x_start = self.CG(A,x_start,y,CG_iteration)

            x_start = 2*x_start-1

            if (time_next < 0)&(last_step_consistency==True):
                img = x_start
                imgs.append(img.detach())
                continue

            #img = self.p_posterior(time_next, time, img, x_start)           
            img = self.p_posterior(time_next, time, img, x_start)

            if return_all_timesteps:
                imgs.append(x_start.detach())

        ret = img if not return_all_timesteps else torch.flip(torch.stack(imgs, dim = 1), dims=(1,))

        ret = self.unnormalize(ret)
        return ret
    
    @torch.inference_mode()
    def I4SB_v3_sample_gN_ky(self, cond, SM, SM_T,measurement, gN, ky, CG_iteration=5, image_size=512, batch_size = 16, return_all_timesteps = False, ddim_sample=False, sampling_timesteps=1000, last_step_consistency=False):
        cond = self.normalize(cond)
        channels = self.channels
        batch, device, total_timesteps,  objective = batch_size, self.device, self.num_timesteps, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = cond[:,0:self.channels,:,:]
        x_N = cond[:,0:self.channels,:,:]
        #imgs = [img.detach()]
        imgs = []
        x_start = None
        ATmeasure = self.back_projection(SM_T, measurement, image_size)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            x_start_prev = x_start
            x_start = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = False)

            if (time_next < 0)&(last_step_consistency==False):
                img = x_start
                imgs.append(img.detach())
                continue                  
            
            
            sigma_N = self.std_fwd[total_timesteps - 1]
            sigma_n_bar = self.std_bwd[time]
            sigma_n = self.std_fwd[time]

            if sigma_n_bar < 1e-4:
                x0e = torch.zeros_like(img)
            else:
                x0e = x_start_prev              
            
            x0e = (x0e+1)/2
            x_start = (x_start+1)/2

            ke = gN*(sigma_n_bar*sigma_n_bar)/(sigma_N*sigma_N)
            
            def A(x, SM=SM, img_size=image_size, ke=ke, ky=ky):
                Ax=self.calc_projection(x, SM)
                
                ATAx=self.back_projection(SM_T,Ax,img_size)
                return (1+ke)*x+ky*ATAx
            y = ke*x0e+x_start+ky*ATmeasure
            x_start = self.CG(A,x_start,y,CG_iteration)

            x_start = 2*x_start-1

            if (time_next < 0)&(last_step_consistency==True):
                img = x_start
                imgs.append(img.detach())
                continue
            
            if self.std_bwd[time] < 1e-4:
                img = self.p_posterior(time_next, time, img, x_start)
            else:
                img = self.ddim_p_posterior(time_next, time, img, cond[:,0:self.channels,:,:], x_start, 0)

            #img = self.p_posterior(time_next, time, img, x_start)

            if return_all_timesteps:
                imgs.append(x_start.detach())

        ret = img if not return_all_timesteps else torch.flip(torch.stack(imgs, dim = 1), dims=(1,))

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def I4SB_sample_gN_ky(self, cond, SM, SM_T,measurement, gN, ky, CG_iteration=5, image_size=512, batch_size = 16, return_all_timesteps = False, ddim_sample=False, sampling_timesteps=1000):
        cond = self.normalize(cond)
        channels = self.channels
        batch, device, total_timesteps,  objective = batch_size, self.device, self.num_timesteps, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = cond[:,0:self.channels,:,:]
        x_N = cond[:,0:self.channels,:,:]
        #imgs = [img.detach()]
        imgs = []
        x_start = None
        ATmeasure = self.back_projection(SM_T, measurement, image_size)
        
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            x_start = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = False)

            if time_next < 0:
                img = x_start
                imgs.append(img.detach())
                continue       
            
            
            
            sigma_N = self.std_fwd[total_timesteps - 1]
            sigma_n_bar = self.std_bwd[time]
            sigma_n = self.std_fwd[time]
            if sigma_n_bar < 1e-4:
                x0e = torch.zeros_like(img)
            else:
                x0e = (sigma_N*sigma_N)*img/(sigma_n_bar*sigma_n_bar)-(sigma_n*sigma_n)*x_N/(sigma_n_bar*sigma_n_bar)
            
            x0e = (x0e+1)/2
            x_start = (x_start+1)/2

            """
            print("before CG")
            projection = self.calc_projection(x_start, SM)
            r = measurement-projection
            rTr = torch.sum(r*r, dim=[2],keepdim=True)
            print(rTr)
            """
            ke = gN*(sigma_n_bar*sigma_n_bar)/(sigma_N*sigma_N)
            
            def A(x, SM=SM, img_size=image_size, ke=ke, ky=ky):
                Ax=self.calc_projection(x, SM)
                
                ATAx=self.back_projection(SM_T,Ax,img_size)
                return (1+ke)*x+ky*ATAx
            y = ke*x0e+x_start+ky*ATmeasure
            x_start = self.CG(A,x_start,y,CG_iteration)
            """
            print("after CG")
            projection = self.calc_projection(x_start, SM)
            r = measurement-projection
            rTr = torch.sum(r*r, dim=[2],keepdim=True)
            print(rTr)
            """
            x_start = 2*x_start-1

            img = self.p_posterior(time_next, time, img, x_start)

            if return_all_timesteps:
                imgs.append(x_start.detach())

        ret = img if not return_all_timesteps else torch.flip(torch.stack(imgs, dim = 1), dims=(1,))

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def I4SB_sample_fx_fy(self, cond, SM, SM_T,measurement, fx, fy, CG_iteration=5, image_size=512, batch_size = 16, return_all_timesteps = False, ddim_sample=False, sampling_timesteps=1000):
        cond = self.normalize(cond)
        channels = self.channels
        batch, device, total_timesteps,  objective = batch_size, self.device, self.num_timesteps, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = cond[:,0:self.channels,:,:]
        x_N = cond[:,0:self.channels,:,:]
        #imgs = [img.detach()]
        imgs = []
        x_start = None
        ATmeasure = self.back_projection(SM_T, measurement, image_size)
        
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            x_start = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = False)

            if time_next < 0:
                img = x_start
                imgs.append(img.detach())
                continue       
            
            
            
            sigma_N = self.std_fwd[total_timesteps - 1]
            sigma_n_bar = self.std_bwd[time]
            sigma_n = self.std_fwd[time]
            if sigma_n_bar < 1e-4:
                x0e = torch.zeros_like(img)
            else:
                x0e = (sigma_N*sigma_N)*img/(sigma_n_bar*sigma_n_bar)-(sigma_n*sigma_n)*x_N/(sigma_n_bar*sigma_n_bar)
            
            x0e = (x0e+1)/2
            x_start = (x_start+1)/2

            """
            print("before CG")
            projection = self.calc_projection(x_start, SM)
            r = measurement-projection
            rTr = torch.sum(r*r, dim=[2],keepdim=True)
            print(rTr)
            """
            ke = fx*(sigma_n_bar*sigma_n_bar*sigma_n)/(sigma_N*sigma_N*sigma_N)
            ky = fy*(sigma_n*sigma_n*sigma_n)/(sigma_N*sigma_N*sigma_N)
            #ky = fy*(time+1)/total_timesteps

            def A(x, SM=SM, img_size=image_size, ke=ke, ky=ky):
                Ax=self.calc_projection(x, SM)
                
                ATAx=self.back_projection(SM_T,Ax,img_size)
                return (1+ke)*x+ky*ATAx
            y = ke*x0e+x_start+ky*ATmeasure
            x_start = self.CG(A,x_start,y,CG_iteration)
            """
            print("after CG")
            projection = self.calc_projection(x_start, SM)
            r = measurement-projection
            rTr = torch.sum(r*r, dim=[2],keepdim=True)
            print(rTr)
            """
            x_start = 2*x_start-1

            img = self.p_posterior(time_next, time, img, x_start)

            if return_all_timesteps:
                imgs.append(x_start.detach())

        ret = img if not return_all_timesteps else torch.flip(torch.stack(imgs, dim = 1), dims=(1,))

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def I4SB_sample_gN(self, cond, SM, SM_T,measurement, gN, kY, CG_iteration=5, image_size=512, batch_size = 16, return_all_timesteps = False, ddim_sample=False, sampling_timesteps=1000):
        cond = self.normalize(cond)
        channels = self.channels
        batch, device, total_timesteps,  objective = batch_size, self.device, self.num_timesteps, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = cond[:,0:self.channels,:,:]
        x_N = cond[:,0:self.channels,:,:]
        #imgs = [img.detach()]
        before_CG_imgs = []
        after_CG_imgs=[]
        x_start = None
        ATmeasure = self.back_projection(SM_T, measurement, image_size)
        
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            x_start = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = False)

            if time_next < 0:
                img = x_start
                before_CG_imgs.append(img.detach())
                continue       
            
            if return_all_timesteps:
                before_CG_imgs.append(x_start.detach())
            
            sigma_N = self.std_fwd[total_timesteps - 1]
            sigma_n_bar = self.std_bwd[time]
            sigma_n = self.std_fwd[time]
            if sigma_n_bar < 1e-4:
                x0e = torch.zeros_like(img)
            else:
                x0e = (sigma_N*sigma_N)*img/(sigma_n_bar*sigma_n_bar)-(sigma_n*sigma_n)*x_N/(sigma_n_bar*sigma_n_bar)
            
            x0e = (x0e+1)/2
            x_start = (x_start+1)/2

            ke = gN*(sigma_n_bar*sigma_n_bar)/(sigma_N*sigma_N)
            ky = kY*(sigma_n*sigma_n)/(sigma_N*sigma_N)
            
            def A(x, SM=SM, img_size=image_size, ke=ke, ky=ky):
                Ax=self.calc_projection(x, SM)
                ATAx=self.back_projection(SM_T,Ax,img_size)
                return (1+ke)*x+ky*ATAx

            y = ke*x0e+x_start+ky*ATmeasure
            x_start = self.CG(A,x_start,y,CG_iteration)
            
            x_start = 2*x_start-1

            img = self.p_posterior(time_next, time, img, x_start)

            if return_all_timesteps:
                after_CG_imgs.append(x_start.detach())

        if not return_all_timesteps:
            return self.unnormalize(img)
        else:
            before_CG_imgs = self.unnormalize(torch.flip(torch.stack(before_CG_imgs, dim = 1), dims=(1,)))
            after_CG_imgs = self.unnormalize(torch.flip(torch.stack(after_CG_imgs, dim = 1), dims=(1,)))
            return [before_CG_imgs, after_CG_imgs]

    @torch.inference_mode()
    def CDDB_sample(self, cond, SM, SM_T, measurement, CG_iteration=5, image_size=512, batch_size = 16, return_all_timesteps = False, ddim_sample=False, sampling_timesteps=1000,last_step_consistency=False):
        cond = self.normalize(cond)
        channels = self.channels
        batch, device, total_timesteps,  objective = batch_size, self.device, self.num_timesteps, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = cond[:,0:self.channels,:,:]
        x_N = cond[:,0:self.channels,:,:]
        #imgs = [img.detach()]
        imgs = []
        x_start = None
        ATmeasure = self.back_projection(SM_T, measurement, image_size)
        
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            x_start = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = False)

            if (time_next < 0)&(last_step_consistency==False):
                img = x_start
                imgs.append(img.detach())
                continue       
           
            def A(x, SM=SM, img_size=image_size):
                Ax=self.calc_projection(x, SM)
                ATAx=self.back_projection(SM_T,Ax,img_size)
                return ATAx
            
            x_start = (x_start+1)/2
            
            y = ATmeasure
            x_start = self.CG(A,x_start,y,CG_iteration)

            x_start = 2*x_start-1

            if (time_next < 0)&(last_step_consistency==True):
                img = x_start
                imgs.append(img.detach())
                continue
            img = self.p_posterior(time_next, time, img, x_start)

            if return_all_timesteps:
                imgs.append(x_start.detach())

        ret = img if not return_all_timesteps else torch.flip(torch.stack(imgs, dim = 1), dims=(1,))

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def CDDB_sample_1step(self, cond, SM, SM_T, measurement, alpha, image_size=512, batch_size = 16, return_all_timesteps = False, ddim_sample=False, sampling_timesteps=1000, last_step_consistency=False):
        cond = self.normalize(cond)
        channels = self.channels
        batch, device, total_timesteps,  objective = batch_size, self.device, self.num_timesteps, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = cond[:,0:self.channels,:,:]
        x_N = cond[:,0:self.channels,:,:]
        #imgs = [img.detach()]
        imgs = []
        x_start = None
        ATmeasure = self.back_projection(SM_T, measurement, image_size)
        
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            x_start = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = False)

            if (time_next < 0)&(last_step_consistency==False):
                img = x_start
                imgs.append(img.detach())
                continue       
            x_start = (x_start+1)/2

            Ax=self.calc_projection(x_start, SM)
            ATAx=self.back_projection(SM_T,Ax,image_size)                          

            y = ATmeasure
            x_start = x_start+alpha*(y-ATAx)

            x_start = 2*x_start-1

            if (time_next < 0)&(last_step_consistency==True):
                img = x_start
                imgs.append(img.detach())
                continue 
            img = self.p_posterior(time_next, time, img, x_start)

            if return_all_timesteps:
                imgs.append(x_start.detach())

        ret = img if not return_all_timesteps else torch.flip(torch.stack(imgs, dim = 1), dims=(1,))

        ret = self.unnormalize(ret)
        return ret

    def CDDB_deep_sample(self, cond, SM, SM_T, measurement, alpha, image_size=512, batch_size = 16, return_all_timesteps = False, ddim_sample=False, sampling_timesteps=1000,last_step_consistency=False):
        cond = self.normalize(cond)
        channels = self.channels
        batch, device, total_timesteps,  objective = batch_size, self.device, self.num_timesteps, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = cond[:,0:self.channels,:,:]
        x_N = cond[:,0:self.channels,:,:]
        #imgs = [img.detach()]
        imgs = []
        x_start = None
        
        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            img = img.requires_grad_()
            
            x_start = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = False)

            if (time_next < 0)&(last_step_consistency==False):
                img = x_start
                img = img.detach()
                imgs.append(img)
                continue       
            x_start = (x_start+1)/2

            Ax=self.calc_projection(x_start, SM)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference,dim=[1,2])
            norm_sum = torch.sum(norm)
            norm_grad = torch.autograd.grad(outputs=norm_sum, inputs=img)[0]
            x_start = x_start-alpha*norm_grad
            x_start = 2*x_start-1
            
            #print(torch.cuda.memory_allocated())
            img = img.detach()
            x_start = x_start.detach()

            if (time_next < 0)&(last_step_consistency==True):
                img = x_start
                img = img.detach()
                imgs.append(img)
                continue
            img = self.p_posterior(time_next, time, img, x_start)

            if return_all_timesteps:
                imgs.append(x_start.detach())

        ret = img if not return_all_timesteps else torch.flip(torch.stack(imgs, dim = 1), dims=(1,))

        ret = self.unnormalize(ret)
        return ret


    @torch.inference_mode()
    def ita_func_ddim_sample(self, cond, ita_func, image_size=128, batch_size = 16, return_all_timesteps = False, ddim_sample=False, sampling_timesteps=1000):
        cond = self.normalize(cond)
        channels = self.channels
        batch, device, total_timesteps,  objective = batch_size, self.device, self.num_timesteps, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = cond[:,0:self.channels,:,:]
        #imgs = [img.detach()]
        imgs = []
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            time_cond = torch.full((batch,), time, device = device, dtype = torch.long)
            self_cond = x_start if self.self_condition else None
            x_start = self.model_predictions(img, cond, time_cond, self_cond, clip_x_start = False)

            if time_next < 0:
                img = x_start
                imgs.append(img.detach())
                continue
            
            if self.std_bwd[time] < 1e-4:
                img = self.p_posterior(time_next, time, img, x_start)
            else:
                print(ita_func(time_next))
                img = self.ddim_p_posterior(time_next, time, img, cond[:,0:self.channels,:,:], x_start, ita_func(time_next))

            if return_all_timesteps:
                imgs.append(x_start.detach())

        ret = img if not return_all_timesteps else torch.flip(torch.stack(imgs, dim = 1), dims=(1,))

        ret = self.unnormalize(ret)
        return ret

    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device = device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2

        x_start = None

        for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, self_cond)

        return img

    @autocast(enabled = False)
    def q_sample(self, step, x0, x1):
        """ Sample q(x_t | x_0, x_1), i.e. eq 11 """

        assert x0.shape == x1.shape
        batch, *xdim = x0.shape

        mu_x0  = unsqueeze_xdim(self.mu_x0[step],  xdim)
        mu_x1  = unsqueeze_xdim(self.mu_x1[step],  xdim)
        std_sb = unsqueeze_xdim(self.std_sb[step], xdim)

        xt = mu_x0 * x0 + mu_x1 * x1
        xt = xt + std_sb * torch.randn_like(xt)
        
        return xt.detach()
    def get_std_fwd(self, step, xdim=None):
        std_fwd = self.std_fwd[step]
        return std_fwd if xdim is None else unsqueeze_xdim(std_fwd, xdim)
    
    def compute_label(self, step, x0, xt):
        """ Eq 12 """
        std_fwd = self.get_std_fwd(step, xdim=x0.shape[1:])
        label = (xt - x0) / std_fwd
        return label.detach()
    
    def p_losses(self, x_start, cond, t, noise = None, offset_noise_strength = None):
        b, c, h, w = x_start.shape 
 
        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device = self.device)
            noise += offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # noise sample
        x = self.q_sample(step=t, x0=x_start, x1=cond[:,0:self.channels, :,:])
        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.inference_mode():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        # predict and take gradient step
        model_out = self.model(x, cond, t, x_self_cond)

        target = self.compute_label(step=t, x0=x_start, xt=x)

        loss = F.mse_loss(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        #loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, cond, residual=False, residual_rescale=5, *args, **kwargs):
                      
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        img = self.normalize(img)
        cond = self.normalize(cond)
        if residual:
            return self.p_losses(residual_rescale*(img-cond[:,0:self.channels,:,:]), cond, t, *args, **kwargs)
            #return self.p_losses(residual_rescale*(img-cond), cond, t, *args, **kwargs)
        return self.p_losses(img, cond, t, *args, **kwargs)

# dataset classes

class HL_test_Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        original_image_size,
        exts = ['raw'],
        position_encoding = False
    ):
        super().__init__()
        self.folder = folder
        self.original_image_size = original_image_size
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        self.position_encoding = position_encoding
        """
        self.imgs = np.zeros((self.__len__(), 2, self.original_image_size, self.original_image_size), "float32")
        for i in range(self.__len__()):
            #print(i)
            path = self.paths[i]
            self.imgs[i,:,:,:] = xc.rawread(path, [2, self.original_image_size, self.original_image_size], "float")
        self.imgs = torch.tensor(self.imgs)
        #self.imgs = self.transform(self.imgs)
        self.imgs = (self.imgs-1000)/2000
        # position encoding
        if position_encoding:
            x = torch.linspace(-1, 1, self.original_image_size)
            y = torch.linspace(-1, 1, self.original_image_size)
            x1, y1 = torch.meshgrid(x, y)
            x1 = x1.repeat([self.__len__(),1,1,1])
            y1 = y1.repeat([self.__len__(),1,1,1])
            self.imgs = torch.cat((self.imgs, x1, y1), dim=1)
            #print(self.imgs.shape)
        """

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        #path = self.paths[index]
        #img = xc.rawread(path, [2, self.original_image_size, self.original_image_size], "float")
        #img = self.imgs[index, :,:,:]
        path = self.paths[index]
        img = xc.rawread(path, [2, self.original_image_size, self.original_image_size], "float")

        img = torch.tensor(img)

        # position encoding
        if self.position_encoding:
            x = torch.linspace(0, 1, self.original_image_size)
            y = torch.linspace(0, 1, self.original_image_size)
            x1, y1 = torch.meshgrid(x, y)
            x1 = x1.repeat([1,1,1])
            y1 = y1.repeat([1,1,1])
            
            img = torch.cat((img, x1, y1), dim=0)
            #print(self.imgs.shape)
        
        i = randint(self.image_size, self.original_image_size)
        j = randint(self.image_size, self.original_image_size)
        img = img[:, i-self.image_size:i, j-self.image_size:j]
        #img = Image.open(path)
        
        return img

class CustomDataset_PET_Denoise_test_2D(torch.utils.data.Dataset):
    def __init__(self,
                 path):
        mr_path = path + "mr/"
        full_dose_path = path + "imgfull/"

        filenames = os.listdir(mr_path)

        self.dataset = []

        pet_noisy=['img4ds','img6ds','img8ds','img10ds'] # 'img10ds' is not seen.

        self.fixer = 8 # from unet model
        self.patchsize=160

        for file_idx in filenames:
            mr=os.path.join(mr_path, file_idx)
            mr = nibabel.load(mr)
            mr = mr.get_fdata()
            mr = mr.astype(np.float32)


            full_dose = os.path.join(full_dose_path, file_idx)
            full_dose = nibabel.load(full_dose)
            full_dose = full_dose.get_fdata()
            full_dose = full_dose.astype(np.float32)

            zpad_mr = self.zpad_data(mr, self.fixer)
            zpad_full_dose = self.zpad_data(full_dose, self.fixer)

            for pet_noisy_item in pet_noisy:
                less_dose_path = path + f"{pet_noisy_item}/"

                less_dose = os.path.join(less_dose_path, file_idx)

                less_dose = nibabel.load(less_dose)
                less_dose = less_dose.get_fdata()
                less_dose = less_dose.astype(np.float32)


                zpad_less_dose=self.zpad_data(less_dose,self.fixer)

                self.dataset.append({"mr": zpad_mr, "full_dose": zpad_full_dose, "less_dose": zpad_less_dose, "name":file_idx[:-7],"dose":pet_noisy_item})

        resolution_x = 160
        resolution_y = 160
        resolution_z = 160

        self.x_pos = np.arange(resolution_x)
        self.x_pos = (self.x_pos / (resolution_x - 1) - 0.5) * 2.

        self.x_pos = self.x_pos[None, :, None, None]
        self.x_pos = np.repeat(self.x_pos, repeats=resolution_y, axis=2)
        self.x_pos = np.repeat(self.x_pos, repeats=resolution_z, axis=3)

        self.y_pos = np.arange(resolution_y)
        self.y_pos = (self.y_pos / (resolution_y - 1) - 0.5) * 2.

        self.y_pos = self.y_pos[None, None, :, None]
        self.y_pos = np.repeat(self.y_pos, repeats=resolution_x, axis=1)
        self.y_pos = np.repeat(self.y_pos, repeats=resolution_z, axis=3)

        self.z_pos = np.arange(resolution_z)
        self.z_pos = (self.z_pos / (resolution_z - 1) - 0.5) * 2.

        self.z_pos = self.z_pos[None, None, None, :]
        self.z_pos = np.repeat(self.z_pos, repeats=resolution_x, axis=1)
        self.z_pos = np.repeat(self.z_pos, repeats=resolution_y, axis=2)


    def zpad_data(self,data_volume,fixer):


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

    def normalize_residual(self,mr,full_dose,less_dose,name,dose):
        o_less_dose=less_dose.copy()
        for z_idx in range (mr.shape[-1]):
            mrslice=mr[:,:,z_idx]

            if (np.max(mrslice)-np.min(mrslice))>0.:
                mr_0, mr_99 = np.percentile(mrslice, 0), np.percentile(mrslice, 99)
                nrm_mr = (mrslice - mr_0) / (mr_99 - mr_0)
                mr[:,:,z_idx] = np.clip(nrm_mr, 0.0, 1.0)

        mins=np.zeros((less_dose.shape[-1]))
        maxs=np.zeros((less_dose.shape[-1]))
        for z_idx in range(less_dose.shape[-1]):
            lessslice = less_dose[:, :, z_idx]
            less_min=np.min(lessslice)
            less_max=np.max(lessslice)
            if (np.max(lessslice) - np.min(lessslice)) > 0.:
                lessslice = (lessslice - less_min) / (less_max -less_min)
                less_dose[:, :, z_idx] = lessslice
                mins[z_idx]=less_min
                maxs[z_idx]=less_max
            else:
                mins[z_idx] = 0.0
                maxs[z_idx] = 0.0



        nrm_mr= np.expand_dims(mr,axis=0)
        nrm_full_dose = np.expand_dims(full_dose, axis=0)
        nrm_less_dose= np.expand_dims(less_dose,axis=0)
        o_less_dose= np.expand_dims(o_less_dose,axis=0)
        return nrm_mr,nrm_full_dose,nrm_less_dose,mins,maxs,name,dose,o_less_dose




    def __getitem__(self, index: int):
        mr,full_dose,less_dose,mins,maxs,name,dose,o_less_dose =self.normalize_residual(self.dataset[index]["mr"],
                            self.dataset[index]["full_dose"],
                            self.dataset[index]["less_dose"],
                            self.dataset[index]["name"],
                            self.dataset[index]["dose"])
        return mr, full_dose, less_dose, mins,maxs,name, dose,o_less_dose


    def __len__(self) -> int:
        return len(self.dataset)


class HL_Dataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        original_image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff', 'npy'],
        augment_horizontal_flip = True,
        convert_image_to = None,
        position_encoding = False
    ):
        super().__init__()
        self.folder = folder
        self.position_encoding = position_encoding

        mr_path = folder + "mr/"
        full_dose_path = folder + "imgfull/"

        filenames = os.listdir(mr_path)
        # filenames=filenames[:2]
        self.dataset = []
        pet_noisy = ['img4ds', 'img6ds', 'img8ds']  # 'img10ds' is not seen.

        self.fixer = 8  # from unet model

        for file_idx in filenames:
            mr = os.path.join(mr_path, file_idx)
            mr = nibabel.load(mr)
            mr = mr.get_fdata()
            mr = mr.astype(np.float32)

            full_dose = os.path.join(full_dose_path, file_idx)
            full_dose = nibabel.load(full_dose)
            full_dose = full_dose.get_fdata()
            full_dose = full_dose.astype(np.float32)

            zpad_mr = self.zpad_data(mr, self.fixer)
            zpad_full_dose = self.zpad_data(full_dose, self.fixer)

            for pet_noisy_item in pet_noisy:
                less_dose_path = folder + f"{pet_noisy_item}/"

                less_dose = os.path.join(less_dose_path, file_idx)

                less_dose = nibabel.load(less_dose)
                less_dose = less_dose.get_fdata()
                less_dose = less_dose.astype(np.float32)

                zpad_less_dose = self.zpad_data(less_dose, self.fixer)

                for z_idx in range(10, zpad_mr.shape[-1] - 10):
                    self.dataset.append({"mr": zpad_mr[:, :, z_idx], "full_dose": zpad_full_dose[:, :, z_idx],
                                         "less_dose": zpad_less_dose[:, :, z_idx]})


    def __len__(self):
        return len(self.dataset)
    def zpad_data(self,data_volume,fixer):


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
    def normalize_residual(self,mr,full_dose,less_dose):

        mr_0, mr_99=np.percentile(mr,0),np.percentile(mr,99)
        nrm_mr= (mr-mr_0)/(mr_99-mr_0)
        nrm_mr=np.clip(nrm_mr,0.0,1.0)


        l_min, l_max = np.min(less_dose), np.max(less_dose)
        nrm_full_dose = (full_dose - l_min) / (l_max - l_min)
        nrm_less_dose = (less_dose - l_min) / (l_max - l_min)


        nrm_mr= np.expand_dims(nrm_mr,axis=0)
        nrm_full_dose = np.expand_dims(nrm_full_dose, axis=0)
        nrm_less_dose= np.expand_dims(nrm_less_dose,axis=0)

        return nrm_mr,nrm_full_dose,nrm_less_dose
    def __getitem__(self, index: int):
        mr,full_dose,less_dose=self.normalize_residual(self.dataset[index]["mr"],
                            self.dataset[index]["full_dose"],
                            self.dataset[index]["less_dose"])


        mr_loss_dose= np.concatenate([full_dose,mr,less_dose],axis=0)
        mr_loss_dose = torch.tensor(mr_loss_dose, dtype=torch.float32)
        return mr_loss_dose

# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 4,
        augment_horizontal_flip = True,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 4,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        convert_image_to = None,
        calculate_fid = False,
        inception_block_idx = 2048,
        max_grad_norm = 1.,
        num_fid_samples = 50000,
        save_best_and_latest_only = False,
        original_image_size = 512,
        residual = False,
        residual_rescale = 5,
        position_encoding = False
    ):
        super().__init__()
        # accelerator
        self.folder = folder
        self.convert_image_to = convert_image_to

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.cond_channels = diffusion_model.model.cond_channels
        self.augment_horizontal_flip = augment_horizontal_flip
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size

        self.gradient_accumulate_every = gradient_accumulate_every
        #assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (train_batch_size x gradient_accumulate_every) should be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.loss_record = torch.zeros(self.train_num_steps)

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader
        self.original_image_size = original_image_size
        #self.dl = dl
        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically
        
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very time consuming."\
                    "Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                num_fid_samples=num_fid_samples,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for `save_best_and_latest_only`."
            self.best_fid = 1e10 # infinite

        self.save_best_and_latest_only = save_best_and_latest_only
        #self.residual = residual & (self.channels==self.cond_channels)
        self.residual = residual
        self.residual_rescale = residual_rescale
        self.position_encoding = position_encoding
    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': __version__,
            'loss record': self.loss_record
            
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        self.loss_record = data["loss record"]
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def load_path(self, path):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(path, map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        self.loss_record = data["loss record"]
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        
        self.ds = HL_Dataset(self.folder, self.image_size, original_image_size=self.original_image_size, augment_horizontal_flip = self.augment_horizontal_flip, convert_image_to = self.convert_image_to, position_encoding=self.position_encoding)

        assert len(self.ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'
        dl = DataLoader(self.ds, batch_size = self.batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())
        
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        accelerator = self.accelerator
        device = accelerator.device
        print(self.residual)
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.
                #data = torch.zeros()
                for _ in range(self.gradient_accumulate_every):
                    
                    img = next(self.dl)
                    data = img[:,0:self.channels,:,:].to(device)
                    cond = img[:, self.channels:self.channels + self.cond_channels, :, :].to(device)

                    del img

                    #data = data.to(device)
                    #cond = cond.to(device)
                    print(cond.shape[0])
                    print(data.shape)
                    #print(data)
                    with self.accelerator.autocast():
                        loss = self.model(data, cond, self.residual, self.residual_rescale)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    
                    self.accelerator.backward(loss)
                self.loss_record[self.step] = total_loss
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        milestone = self.step // self.save_and_sample_every
                        self.ema.ema_model.eval()
                        if cond.shape[0] >= self.num_samples:
                            with torch.inference_mode():   
                                #batches = num_to_groups(self.num_samples, self.batch_size)
                                #all_images_list = list(map(lambda n: self.ema.ema_model.sample(batch_size=n), batches))

                                input=cond[0:self.num_samples,:,:,:]
                                sample_images = self.ema.ema_model.sample(input, batch_size=self.num_samples)


                            if self.residual:
                                utils.save_image(((sample_images-0.5)/self.residual_rescale)+cond[0:self.num_samples,0:self.channels,:,:], str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                            else:
                                utils.save_image(sample_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                            utils.save_image(data[0:self.num_samples,:,:,:], str(self.results_folder / f'HR-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                            utils.save_image(cond[0:self.num_samples,0:1,:,:], str(self.results_folder / f'MR-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))
                            utils.save_image(cond[0:self.num_samples,1:2,:,:], str(self.results_folder / f'LR-{milestone}.png'), nrow = int(math.sqrt(self.num_samples)))

                        # whether to calculate fid

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            accelerator.print(f'fid_score: {fid_score}')
                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
