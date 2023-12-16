"""Generate images using pretrained network pickle."""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "4"
from typing import List, Optional

import click
from numpy.lib.type_check import imag
import dnnlib
import numpy as np
import torch
import tempfile
import legacy
import random

from training.data.pred_loader import GarmentDataset, GarmentTestDataset

import warnings
warnings.filterwarnings("ignore")
from colorama import init
from colorama import Fore, Style
from icecream import ic
init(autoreset=True)
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_utils import training_stats
from torch_utils import custom_ops
import copy
import pandas as pd

from metrics.evaluation.evaluator import GarmentEvaluator
from metrics.evaluation.Garment_data import PrecomputedGarmentResultsDataset as PrecomputedInpaintingResultsDataset
from metrics.evaluation.losses.base_loss import SSIMScore, LPIPSScore, FIDScore, c_FIDScore, c_LPIPSScore
from metrics.evaluation.utils import load_yaml

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

def visualize_gen(i, comp_img):
    lo, hi = [-1, 1]

    comp_img = np.asarray(comp_img[0].cpu(), dtype=np.float32).transpose(1, 2, 0)
    comp_img = (comp_img - lo) * (255 / (hi - lo))
    comp_img = np.rint(comp_img).clip(0, 255).astype(np.uint8)
    plt.imsave(f'fid_gens/' + i + '000.png', comp_img / 255)
    plt.close()


def create_folders():
    if not os.path.exists(f'fid_gens/'):
        os.makedirs(f'fid_gens/')


def save_gen(G, rank, num_gpus, device, img_data, resolution, label, truncation_psi):
    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)
    dataset = GarmentTestDataset(img_data, resolution)# dataset = GarmentDataset(img_data, resolution)
    num_items = len(dataset)

    data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)
    # Main loop.
    item_subset = [(i * num_gpus + rank) % num_items for i in range((num_items - 1) // num_gpus + 1)]
    dataloader = torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=1, **data_loader_kwargs)
    for _, data in tqdm(enumerate(dataloader, 0), total=len(dataloader), desc=f'Visualizing on GPU: {rank}'):

        with torch.no_grad():
            ## data is a tuple of (rgbs, rgbs_erased, amodal_tensors, visible_mask_tensors, erased_modal_tensors) ####
            texture, sketch, fnames = data # img, texture, sketch, fnames = data
            sketch = sketch.to(device)
            texture = texture.to(device)

            fname = fnames[0]
            comp_img = G(img=torch.cat([sketch, texture], dim=1), c=label, truncation_psi=truncation_psi, noise_mode='const')
            visualize_gen(fname, comp_img.detach())

def run(rank, num_gpus, temp_dir, img_data, pred_dir, get_mask=True, get_original_texture=True):
    # Init torch.distributed.
    if num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    custom_ops.verbosity = 'none'

    # Print network summary.
    device = torch.device('cuda', rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    eval_config = load_yaml('metrics/configs/eval2_gpu.yaml')


    if rank == 0:
        eval_dataset = PrecomputedInpaintingResultsDataset(img_data, f'{pred_dir}/',
                                                           get_mask=get_mask,
                                                           get_original_texture=get_original_texture)

        metrics = {
            'ssim': SSIMScore(),
            'lpips': LPIPSScore(),
            'fid': FIDScore(),
            'c_fid':c_FIDScore(),
            'c_lpips':c_LPIPSScore()
        }
        evaluator = GarmentEvaluator(eval_dataset, scores=metrics, area_grouping=False,
                                integral_title='lpips_fid100_f1', integral_func=None,
                                use_mask=get_mask, use_texture=get_original_texture,
                                **eval_config.evaluator_kwargs)
        results = evaluator.dist_evaluate(device, num_gpus=1, rank=0)
        results = pd.DataFrame(results).stack(1).unstack(0)
        results.dropna(axis=1, how='all', inplace=True)
        results.to_csv(f'{pred_dir}/eval_result.csv', sep='\t', float_format='%.4f')
        print(results)
    

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--img_data', help='Training images (directory)', metavar='PATH', required=True)
@click.option('--pred_dir', help='Training images (directory)', metavar='PATH', required=True)
@click.option('--resolution', help='Res of Images [default: 256]', type=int, metavar='INT')
@click.option('--num_gpus', help='Number of gpus [default: 1]', type=int, metavar='INT')

def generate_images(
    ctx: click.Context,
    truncation_psi: float,
    img_data: str,
    pred_dir: str,
    resolution: int,
    num_gpus: int,
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
    
    if num_gpus is None:
        num_gpus = 1

    if not num_gpus >= 1:
        ctx.fail('--gpus must be at least 1')

    
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if num_gpus == 1:
            run(rank=0, num_gpus=num_gpus, temp_dir=temp_dir, img_data=img_data, pred_dir=pred_dir)
        else:
            torch.multiprocessing.spawn(fn=run_gen, args=(num_gpus, temp_dir, G, img_data, resolution, label, truncation_psi), nprocs=num_gpus)


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
