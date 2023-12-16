import random

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix
from icecream import ic

from .fft_loss import Frequency_PercLoss, Frequency_AutocorrLoss, define_VGGF16
from .high_receptive_pl import HRFPL
import os


# ----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, sync, gain):  # to be overridden by subclass
        raise NotImplementedError()


# ----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G_encoder, G_sketch_encoder, G_scft, G_mapping, G_synthesis, D, augment_pipe=None, style_mixing_prob=0.9,
                 r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2):
        super().__init__()
        self.device = device
        self.G_encoder = G_encoder
        self.G_sketch_encoder = G_sketch_encoder
        self.G_scft = G_scft
        self.G_mapping = G_mapping
        self.G_synthesis = G_synthesis
        self.D = D
        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)

        self.hrfpl_gama = 5  # defult: 5
        if self.hrfpl_gama > 0:
            self.run_hrfpl = HRFPL(weight=self.hrfpl_gama, weights_path=os.getcwd())

        # frequency perceptual loss args.
        self.fp_gama = 4
        self.fp_layers = ['8', '11', '13']  # train2: ['20', '22', '25', '27']; train3:['8', '11', '13']
        self.vgg16 = define_VGGF16()
        self.fqloss_patchnum = 5
        if self.fp_gama > 0:
            self.fploss = Frequency_PercLoss(vgg_module=self.vgg16, select_layers=self.fp_layers,
                                             frequency_mode="RGB", what_to_cul="amplitude")

        self.gobal_fp_gama = 0
        if self.gobal_fp_gama > 0:
            self.gobal_fploss = Frequency_PercLoss(vgg_module=self.vgg16, select_layers=self.fp_layers,
                                                   frequency_mode="RGB")

        # local frequency autocorrLoss args
        self.fa_gama = 0
        self.fa_layers = ['8', '11', '13']
        self.faloss_patchnum = 3
        if self.fa_gama > 0:
            self.faloss = Frequency_AutocorrLoss(select_layers=self.fa_layers)


    def run_G(self, r_img, c, sync):
        with misc.ddp_sync(self.G_encoder, sync):
            img_sketch = r_img[:, 0:3, :, :]
            img_texture = r_img[:, 3:6, :, :]
            x_global, z, feats = self.G_encoder(img_texture, c)
            x_sketch, sketch_feats = self.G_sketch_encoder(img_sketch, c)
            # x_global + x_sketch  # add features
            x_global = self.G_scft(x_sketch, x_global)

        with misc.ddp_sync(self.G_mapping, sync):
            ws = self.G_mapping(z, c)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff,
                                         torch.full_like(cutoff, ws.shape[1]))

                    # for single-gpu (1 line)

                    ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
                    """
                    # for multi-gpu :change 'ws[:, cutoff:] = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]'
                    # to the 4 lines under
                    output = self.G_mapping(torch.randn_like(z), c, skip_w_avg_update=True)
                    ws_pai = ws.clone()
                    ws_pai[:, cutoff:] = output[:, cutoff:]
                    ws = ws_pai
                    """

        with misc.ddp_sync(self.G_synthesis, sync):
            img = self.G_synthesis(x_global, feats, sketch_feats, ws)
        return img, ws

    def run_D(self, img, c, sync):
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def get_texture_mask(self, mask, encode_size):
        midsize = mask.size(-1)//2
        for i in range(midsize):
            left = midsize - i - 1
            if torch.mean(mask[:,:, midsize - 1, left]) / 255 < 0.5:
                left = left
                break
        for i in range(midsize):
            right = midsize + i
            if torch.mean(mask[:, :, midsize, right]) / 255 < 0.5:
                right = right - encode_size
                break
        for i in range(midsize):
            up = midsize - i - 1
            if torch.mean(mask[:,:, up, midsize - 1]) / 255 < 0.5:
                up = up + 40
                break
        for i in range(midsize):
            bottom = midsize + i
            if torch.mean(mask[:,:, bottom, midsize]) / 255 < 0.5:
                bottom = bottom - encode_size
                break
        if right - left > 0 and bottom - up > 0:
            # texture_mask = np.zeros_like(mask)
            # texture_mask[:, left: right, up: bottom] = 255
            texture_mask = {'up': up,
                            'bottom': bottom,
                            'left': left,
                            'right': right}
        else:
            texture_mask = None
        return texture_mask

    def run_local_fftloss(self, image, target, mask, loss_fn, patch_num, use_texture_mask=False, use_same_size=True):
        deafult_encode_size = 64
        max_encode_size = 96
        image_size = target.size(-1)
        batch_size = image.size(0)
        encode_size = random.randint(deafult_encode_size, max_encode_size)
        if not use_same_size:
            encode_size_target = random.randint(deafult_encode_size, encode_size)

        loss = 0
        max_loop = 500

        patch_images_block = None
        patch_targets_block = None
        texture_mask = None

        patch_getted = False

        for b in range(batch_size):
            if use_texture_mask:
                texture_mask = self.get_texture_mask(mask, encode_size)

            for _ in range(patch_num):
                for i in range(max_loop):
                    patch_getted = False

                    if texture_mask is not None:
                        min_left = max(0, texture_mask['left'])
                        max_right = min(image_size - encode_size - 1, texture_mask['right'])
                        min_up = max(0, texture_mask['up'])
                        max_bottom = min(image_size - encode_size - 1, texture_mask['bottom'])
                        if min_left > max_right or min_up > max_bottom:
                            x = (image_size - encode_size) // 2
                            y = (image_size - encode_size) // 2
                        else:
                            x = random.randint(min_up, max_bottom)
                            y = random.randint(min_left, max_right)
                    else:
                        x = random.randint(0, image_size - encode_size - 1)
                        y = random.randint(0, image_size - encode_size - 1)
                    mask_crop = mask[b, :, x: x + encode_size, y:y + encode_size]
                    if 0.98 <= torch.mean(mask_crop) <= 1:
                        if use_same_size:
                            patch_image = image[b, :, x: x + encode_size, y:y + encode_size].unsqueeze(0)
                            patch_target = target[b, :, x: x + encode_size, y:y + encode_size].unsqueeze(0)
                        else:

                            patch_image = image[b, :, x: x + encode_size, y:y + encode_size].unsqueeze(0)
                            patch_target = target[b, :, x: x + encode_size_target, y:y + encode_size_target].unsqueeze(
                                0)
                        # loss += self.fploss(patch_image, patch_target)
                        if patch_images_block is None:
                            patch_images_block = patch_image
                            patch_targets_block = patch_target
                        else:
                            patch_images_block = torch.cat([patch_images_block, patch_image], dim=0)
                            patch_targets_block = torch.cat([patch_targets_block, patch_target], dim=0)
                        patch_getted = True
                        break

                if not patch_getted:
                    clip_start_index = (image_size - encode_size) // 2
                    clip_end_index = clip_start_index + encode_size
                    patch_image = image[b, :, clip_start_index: clip_end_index,
                                  clip_start_index: clip_end_index].unsqueeze(0)
                    if not use_same_size:
                        clip_start_index = (image_size - encode_size_target) // 2
                        clip_end_index = clip_start_index + encode_size_target
                    patch_target = target[b, :, clip_start_index: clip_end_index,
                                   clip_start_index: clip_end_index].unsqueeze(0)

                    if patch_images_block is None:
                        patch_images_block = patch_image
                        patch_targets_block = patch_target
                    else:
                        patch_images_block = torch.cat([patch_images_block, patch_image], dim=0)
                        patch_targets_block = torch.cat([patch_targets_block, patch_target], dim=0)
                    # loss += self.fploss(patch_image, patch_target)

        for i in range(patch_num):
            patch_images = patch_images_block[i * batch_size: (i + 1) * batch_size, :, :, :]
            patch_targets = patch_targets_block[i * batch_size: (i + 1) * batch_size, :, :, :]
            loss += loss_fn(patch_images, patch_targets)

        return loss / (batch_size * patch_num)

    def accumulate_gradients(self, phase, real_img, sketch_img, texture_img, mask, real_c, gen_c, sync, gain):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        do_Gmain = (phase in ['Gmain', 'Gboth'])
        do_Dmain = (phase in ['Dmain', 'Dboth'])
        do_Dr1 = (phase in ['Dreg', 'Dboth']) and (self.r1_gamma != 0)

        # Gmain: Maximize logits for generated images.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                g_inputs = torch.cat([sketch_img, texture_img], dim=1)
                gen_img, _ = self.run_G(g_inputs, gen_c, sync=sync)  # May get synced by Gpl.
                # gen_img = gen_img * mask + real_img * (1 - mask)
                loss_rec = 10 * torch.nn.functional.l1_loss(gen_img, real_img)

                if self.hrfpl_gama > 0:
                    loss_pl = self.run_hrfpl(gen_img, real_img)

                if self.augment_pipe is not None:
                    gen_img = self.augment_pipe(gen_img)
                # d_inputs = torch.cat([0.5 - mask, gen_img], dim=1)
                d_inputs = gen_img
                gen_logits = self.run_D(d_inputs, gen_c, sync=False)

                loss_G = torch.nn.functional.softplus(-gen_logits)  # -log(sigmoid(gen_logits))
                loss_Gmain = loss_G.mean() + loss_rec

                if self.hrfpl_gama > 0:
                    loss_Gmain += loss_pl

                if self.fp_gama > 0:
                    loss_fp = self.run_local_fftloss(gen_img, real_img, mask,
                                                     loss_fn=self.fploss,
                                                     patch_num=self.fqloss_patchnum,
                                                     use_texture_mask=True,
                                                     use_same_size=False) * self.fp_gama
                    loss_Gmain += loss_fp

                if self.fa_gama > 0:
                    loss_fa = self.run_local_fftloss(gen_img, real_img, mask,
                                                    loss_fn = self.faloss,
                                                    patch_num=self.fqloss_patchnum,
                                                    use_texture_mask=True,
                                                    use_same_size=True) * self.fa_gama
                    loss_Gmain += loss_fa

                if self.gobal_fp_gama > 0:
                    loss_gbfp = self.gobal_fploss(gen_img, real_img) * self.gobal_fp_gama
                    loss_Gmain += loss_gbfp

                training_stats.report('Loss/G/loss', loss_G)
                training_stats.report('Loss/G/rec_loss', loss_rec)
                training_stats.report('Loss/G/main_loss', loss_Gmain)

                if self.hrfpl_gama > 0:
                    training_stats.report('Loss/G/pl_loss', loss_pl)
                if self.fp_gama > 0:
                    training_stats.report('Loss/G/fft_loss', loss_fp)
                if self.gobal_fp_gama > 0:
                    training_stats.report('Loss/G/gb_fft_loss', loss_gbfp)
                if self.fa_gama > 0:
                    training_stats.report('Loss/G/fa_loss', loss_fa)
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                g_inputs = torch.cat([sketch_img, texture_img], dim=1)
                gen_img, _ = self.run_G(g_inputs, gen_c, sync=sync)  # May get synced by Gpl.
                # gen_img = gen_img * mask + real_img * (1 - mask)
                if self.augment_pipe is not None:
                    gen_img = self.augment_pipe(gen_img)
                d_inputs = gen_img

                gen_logits = self.run_D(d_inputs, gen_c, sync=False)  # Gets synced by loss_Dreal.
                loss_Dgen = torch.nn.functional.softplus(gen_logits)  # -log(1 - sigmoid(gen_logits))

            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(do_Dr1)
                if self.augment_pipe is not None:
                    real_img_tmp = self.augment_pipe(real_img_tmp)
                d_inputs = real_img_tmp
                real_logits = self.run_D(d_inputs, real_c, sync=sync)

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)  # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = \
                        torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True,
                                            only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1, 2, 3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

# ----------------------------------------------------------------------------
