import os
import random

import numpy as np
import PIL.Image
import json
import torch
import dnnlib

from . import mask_generator
import albumentations as A

try:
    import pyspng
except ImportError:
    pyspng = None


# ----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 name,  # Name of the dataset.
                 raw_shape,  # Shape of the raw image data (NCHW).
                 max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
                 xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
                 random_seed=0,  # Random seed to use when applying max_size.
                 ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self):  # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


# ----------------------------------------------------------------------------

class GarmentDataset(Dataset):

    def __init__(self,
                 img_path,  # Path to images.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self.sz = resolution
        self.train_path = img_path
        self.img_path = self.train_path + "/image/"
        self.sketch_path = self.train_path + "/sketch/"
        self.mask_path = self.train_path + "/mask/"
        self._type = 'dir'
        self.files = []

        self._all_image_fnames = [os.path.relpath(os.path.join(root, fname), start=self.img_path) for root, _dirs, files
                                  in
                                  os.walk(self.img_path) for fname in files]
        PIL.Image.init()
        self._image_fnames = sorted(os.path.join(self.img_path, fname) for fname in self._all_image_fnames if
                                    self._file_ext(fname) in PIL.Image.EXTENSION)
        self._fnames = sorted(fname for fname in self._all_image_fnames if
                              self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        self.files = []

        for f in self._fnames:
            self.files.append(f)

        self.files = sorted(self.files)

        self.transform = A.Compose([
            A.PadIfNeeded(min_height=self.sz, min_width=self.sz),
            A.OpticalDistortion(),
            A.RandomCrop(height=self.sz, width=self.sz),
            A.CLAHE(),
            A.ToFloat()
        ])

        name = os.path.splitext(os.path.basename(self.train_path))[0]
        raw_shape = [len(self.files)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def __len__(self):
        return len(self.files)

    def _load_image(self, fn):
        return PIL.Image.open(fn).convert('RGB')

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        image = np.array(PIL.Image.open(fname).convert('RGB'))
        image = self.transform(image=image)['image']
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW

        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_image_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def _get_image(self, idx):
        fname = self.files[idx]
        sketch_fname = os.path.join(self.sketch_path, fname)
        mask_fname = os.path.join(self.mask_path, fname)
        image_fname = os.path.join(self.img_path, fname)

        rgb = np.array(self._load_image(image_fname))  # uint8
        rgb = self.transform(image=rgb)['image']
        rgb = np.rint(rgb * 255).clip(0, 255).astype(np.uint8)

        mask = np.array(self._load_image(mask_fname))  # uint8
        mask = self.transform(image=mask)['image']
        mask = np.rint(mask * 255).clip(0, 255).astype(np.uint8)
        mask[mask > 235] = 255
        mask[mask <= 235] = 0
        mask = np.expand_dims(mask[:, :, 0], axis=0)

        sketch = np.array(self._load_image(sketch_fname))  # uint8
        sketch = self.transform(image=sketch)['image']
        sketch = np.rint(sketch * 255).clip(0, 255).astype(np.uint8)

        texture_mask = self._get_texture_mask(mask)
        texture = self._get_texture(rgb, mask, texture_mask)
        # texture = self._get_texture(rgb, mask)
        return rgb, mask, sketch, texture

    def _get_texture_mask(self, mask):
        midsize = mask.shape[-1]//2
        for i in range(midsize):
            left = midsize - i - 1
            if np.mean(mask[:, midsize - 1, left]) / 255 < 0.5:
                break
        for i in range(midsize):
            right = midsize + i
            if np.mean(mask[:, midsize, right]) / 255 < 0.5:
                right = right - 64
                break
        for i in range(midsize):
            up = midsize - i - 1
            if np.mean(mask[:, up, midsize - 1]) / 255 < 0.5:
                up = up + 40
                break
        for i in range(midsize):
            bottom = midsize + i
            if np.mean(mask[:, bottom, midsize]) / 255 < 0.5:
                bottom = bottom - 64
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

    def _get_texture(self, groundtruth, mask, texture_mask=None):
        deafult_encode_size = 64
        max_encode_size = 96

        image_size = groundtruth.shape[0]

        max_loop = 1000
        for i in range(max_loop):
            encode_size = random.randint(deafult_encode_size, max_encode_size)
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
            mask_crop = mask[:, x: x + encode_size, y:y + encode_size]
            if 0.98 <= np.mean(mask_crop) / 255 <= 1:
                patch = groundtruth[x: x + encode_size, y:y + encode_size, :]
                return patch

        # avoid die
        # center crop
        clip_start_index = (image_size - deafult_encode_size) // 2
        clip_end_index = clip_start_index + deafult_encode_size
        patch = groundtruth[clip_start_index: clip_end_index, clip_start_index: clip_end_index, :]
        return patch

    def _put_patch_center(self, texture, img):
        image_size = img.shape[0]
        texture_size = texture.shape[0]
        start_index = (image_size - texture_size) // 2
        end_index = start_index + texture_size
        img[start_index: end_index, start_index: end_index, :] = texture
        return img

    def __getitem__(self, idx):
        rgb, mask, sketch, texture = self._get_image(idx)  # modal, uint8 {0, 1}

        texture_img = np.zeros_like(rgb)
        texture_img = self._put_patch_center(texture, texture_img)

        rgb = rgb.transpose(2, 0, 1)
        sketch = sketch.transpose(2, 0, 1)
        texture_img = texture_img.transpose(2, 0, 1)
        mask = np.array(mask, dtype=np.float32)

        return rgb, mask, sketch, texture_img, super().get_label(idx)

