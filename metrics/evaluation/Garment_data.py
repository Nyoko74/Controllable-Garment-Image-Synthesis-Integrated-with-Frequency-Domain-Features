import glob
import os

import cv2
import PIL.Image as Image
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import PIL.Image

def load_image(fname, mode='RGB', return_orig=False):
    img = np.array(Image.open(fname).convert(mode))
    if img.ndim == 3:
        img = np.transpose(img, (2, 0, 1))
    out_img = img.astype('float32') / 255
    if return_orig:
        return out_img, img
    else:
        return out_img


def ceil_modulo(x, mod):
    if x % mod == 0:
        return x
    return (x // mod + 1) * mod


def pad_img_to_modulo(img, mod):
    channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return np.pad(img, ((0, 0), (0, out_height - height), (0, out_width - width)), mode='symmetric')


def pad_tensor_to_modulo(img, mod):
    batch_size, channels, height, width = img.shape
    out_height = ceil_modulo(height, mod)
    out_width = ceil_modulo(width, mod)
    return F.pad(img, pad=(0, out_width - width, 0, out_height - height), mode='reflect')


def scale_image(img, factor, interpolation=cv2.INTER_AREA):
    if img.shape[0] == 1:
        img = img[0]
    else:
        img = np.transpose(img, (1, 2, 0))

    img = cv2.resize(img, dsize=None, fx=factor, fy=factor, interpolation=interpolation)

    if img.ndim == 2:
        img = img[None, ...]
    else:
        img = np.transpose(img, (2, 0, 1))
    return img


class GarmentDataset(Dataset):
    def __init__(self, datadir,  pad_out_to_modulo=None,get_mask=False, get_original_texture=False, scale_factor=None):
        self.datadir = datadir
        self.get_mask = get_mask
        self.get_original_texture = get_original_texture

        self.sketch_filenames = sorted(list(glob.glob(os.path.join(self.datadir,'sketch','**.jpg'), recursive=True)))
        self.img_filenames =sorted(list(glob.glob(os.path.join(self.datadir,'image','**.jpg'), recursive=True)))
        if self.get_mask == True:
            self.mask_filenames = sorted(list(glob.glob(os.path.join(self.datadir, 'mask', '**.jpg'), recursive=True)))
        if self.get_original_texture == True:
            self.texture_filenames = sorted(list(glob.glob(os.path.join(self.datadir, 'texture', '**.jpg'), recursive=True)))
        self.pad_out_to_modulo = pad_out_to_modulo
        self.scale_factor = scale_factor

    def _put_patch_center(self, texture, img):
        image_size = img.shape[-1]
        texture_size = texture.shape[-1]
        start_index = (image_size - texture_size) // 2
        end_index = start_index + texture_size
        img[:, start_index: end_index, start_index: end_index] = texture
        return img


    def __len__(self):
        return len(self.sketch_filenames)

    def __getitem__(self, i):
        image = load_image(self.img_filenames[i], mode='RGB')
        sketch = load_image(self.sketch_filenames[i], mode='RGB')
        result = dict(image=image, sketch=sketch)
        if self.get_mask:
            mask = load_image(self.mask_filenames[i], mode='RGB')
            result['mask'] = mask
        if self.get_original_texture:
            texture = load_image(self.texture_filenames[i], mode='RGB')
            texture_img = np.zeros_like(image)
            texture_img = self._put_patch_center(texture, texture_img)
            result['texture'] = texture_img
            result['texture_size'] = texture.shape[-1]


        if self.scale_factor is not None:
            result['image'] = scale_image(result['image'], self.scale_factor)
            result['sketch'] = scale_image(result['sketch'], self.scale_factor, interpolation=cv2.INTER_NEAREST)
            if self.get_mask:
                result['mask'] = scale_image(result['mask'], self.scale_factor, interpolation=cv2.INTER_NEAREST)
            if self.get_original_texture:
                result['texture'] = scale_image(result['texture'], self.scale_factor, interpolation=cv2.INTER_NEAREST)

        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['image'] = pad_img_to_modulo(result['image'], self.pad_out_to_modulo)
            result['sketch'] = pad_img_to_modulo(result['sketch'], self.pad_out_to_modulo)
            if self.get_mask:
                result['mask'] = pad_img_to_modulo(result['mask'], self.pad_out_to_modulo)
            if self.get_original_texture:
                result['texture'] = pad_img_to_modulo(result['texture'], self.pad_out_to_modulo)

        return result


class PrecomputedGarmentResultsDataset(GarmentDataset):
    def __init__(self, datadir, predictdir, get_mask=False, get_original_texture=False, **kwargs):
        super().__init__(datadir, get_mask=True, get_original_texture=True, **kwargs)
        if not datadir.endswith('/'):
            datadir += '/'
        self.get_mask = get_mask
        self.get_original_texture = get_original_texture
        self.predictdir = predictdir
        self._all_image_fnames= [os.path.relpath(os.path.join(root, fname), start=self.predictdir) for root, _dirs, files in
                            os.walk(self.predictdir) for fname in files]
        PIL.Image.init()
        self.pred_filenames = sorted(os.path.join(self.predictdir, fname) for fname in self._all_image_fnames if
                                    self._file_ext(fname) in PIL.Image.EXTENSION)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __getitem__(self, i):
        result = super().__getitem__(i)
        result['inpainted'] = load_image(self.pred_filenames[i])
        if self.pad_out_to_modulo is not None and self.pad_out_to_modulo > 1:
            result['inpainted'] = pad_img_to_modulo(result['inpainted'], self.pad_out_to_modulo)
        return result



