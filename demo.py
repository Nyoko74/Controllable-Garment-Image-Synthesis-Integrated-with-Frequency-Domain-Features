"""Generate images using pretrained network pickle."""

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
import re
from typing import List, Optional

import click
from numpy.lib.type_check import imag
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

from training.data.demo_loader import get_loader

import warnings
warnings.filterwarnings("ignore")
from colorama import init
from colorama import Fore, Style
from icecream import ic
init(autoreset=True)
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random

#----------------------------------------------------------------------------

def save_image(img, name):
    x = denormalize(img.detach().cpu())
    x = x.permute(1, 2, 0).numpy()
    x = np.rint(x) / 255.
    plt.imsave(name+'.png', x)
    
def denormalize(tensor):
    pixel_mean = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)
    pixel_std = torch.Tensor([127.5, 127.5, 127.5]).view(3, 1, 1)
    denormalizer = lambda x: torch.clamp((x * pixel_std) + pixel_mean, 0, 255.)

    return denormalizer(tensor)

def visualize_demo(i, sketches, textures, pred_img):
    lo, hi = [-1, 1]

    sketches = np.asarray(sketches[0].cpu(), dtype=np.float32).transpose(1, 2, 0)
    sketches = (sketches - lo) * (255 / (hi - lo))
    sketches = np.rint(sketches).clip(0, 255).astype(np.uint8)

    textures = np.asarray(textures[0].cpu(), dtype=np.float32).transpose(1, 2, 0)
    textures = (textures - lo) * (255 / (hi - lo))
    textures = np.rint(textures).clip(0, 255).astype(np.uint8)

    pred_img = np.asarray(pred_img[0].cpu(), dtype=np.float32).transpose(1, 2, 0)
    pred_img = (pred_img - lo) * (255 / (hi - lo))
    pred_img = np.rint(pred_img).clip(0, 255).astype(np.uint8)

    
    plt.imsave('visualizations/sketches/' + i + '.png', sketches / 255)
    plt.imsave('visualizations/textures/' + i + '.png', textures / 255)
    plt.imsave('visualizations/pred_img/' + i + '.png', pred_img / 255)
    plt.close()

def create_folders():
    if not os.path.exists('visualizations/pred_img/'):
        os.makedirs('visualizations/pred_img/')
    if not os.path.exists('visualizations/sketches/'):
        os.makedirs('visualizations/sketches/')
    if not os.path.exists('visualizations/textures/'):
        os.makedirs('visualizations/textures/')
    if not os.path.exists('visualizations/results/'):
        os.makedirs('visualizations/results/')


def save_image_grid(texture, sketch, pred_img, fname, drange=[-1,1], grid_size=(1,1)):
    lo, hi = (0, 255)

    model_lo, model_hi = drange

    texture = np.asarray(texture, dtype=np.float32)
    texture = (texture - model_lo) * (255 / (model_hi - model_lo))
    texture = np.rint(texture).clip(0, 255).astype(np.uint8)

    sketch = np.asarray(sketch, dtype=np.float32)
    sketch = (sketch - model_lo) * (255 / (model_hi - model_lo))
    sketch = np.rint(sketch).clip(0, 255).astype(np.uint8)

    pred_img = np.asarray(pred_img, dtype=np.float32)
    pred_img = (pred_img - model_lo) * (255 / (model_hi - model_lo))
    pred_img = np.rint(pred_img).clip(0, 255).astype(np.uint8)

    # comp_img = img * (1 - inv_mask) + pred_img * inv_mask
    f_img = np.concatenate((texture, sketch, pred_img), axis=1)

    gw, gh = grid_size
    gw *= f_img.shape[1] // 3
    _N, C, H, W = sketch.shape
    f_img = f_img.reshape(gh, gw, C, H, W)
    f_img = f_img.transpose(0, 3, 1, 4, 2)
    f_img = f_img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    fname = 'visualizations/results/' + fname + '.png'
    if C == 1:
        PIL.Image.fromarray(f_img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(f_img, 'RGB').save(fname)


#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--img_data', help='Training images (directory)', metavar='PATH', required=True)
@click.option('--resolution', help='Res of Images [default: 256]', type=int, metavar='INT')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    img_data: str,
    resolution: int,
    class_idx: Optional[int],
):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if resolution is None:
        resolution = 256

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'] # type: ignore
    
    G = G.eval().to(device)
    
    dataloader = get_loader(img_path=img_data, resolution=resolution)
    ic(G.encoder.b256.img_channels)
    
    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            ctx.fail('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')
    
    netG_params = sum(p.numel() for p in G.parameters())
    print(Fore.BLUE +"Generator Params: {} M".format(netG_params/1e6))
    print(Style.BRIGHT + Fore.GREEN + "Starting Visualization...")
    times = []

    create_folders()
    j = 0
    
    for i, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc='Visualizing..'):

        with torch.no_grad():
            ## data is a tuple of (rgbs, rgbs_erased, amodal_tensors, visible_mask_tensors, erased_modal_tensors) ####
            textures, sketches, fnames = data

            textures = textures.to(device)
            sketches = sketches.to(device)
            fname = fnames[0]

            start_time = time.time()
            pred_img = G(img=torch.cat([sketches, textures], dim=1), c=label, truncation_psi=truncation_psi, noise_mode='const')

            end_time = time.time()
            
            times.append(end_time - start_time)
            
            if torch.mean(sketches).item() != 0:
                j += 1
                visualize_demo(fname, sketches, textures, pred_img.detach())
                save_image_grid(textures.cpu(), sketches.cpu(), pred_img.detach().cpu(),
                                fname, drange=[-1, 1])

    avg_time = np.mean(times)
    print(Fore.CYAN + "Duration per image: {} s".format(avg_time))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
