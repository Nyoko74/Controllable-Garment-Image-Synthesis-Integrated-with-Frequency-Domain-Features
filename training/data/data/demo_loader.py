from tabnanny import filename_only
import numpy as np
import cv2
import os
import PIL
import torch
from .dataset import Dataset


class ImageDataset(Dataset):
    
    def __init__(self,
        img_path,                   # Path to images.
        resolution      = None,     # Ensure specific resolution, None = highest available.
        **super_kwargs,             # Additional arguments for the Dataset base class.
    ):
        self.sz = resolution
        self.img_path = img_path
        self._type = 'dir'
        self.files = []
        self.idx = 0

        self._all_fnames = [os.path.relpath(os.path.join(root, fname), start=self.img_path) for root, _dirs, files in os.walk(self.img_path) for fname in files]
        PIL.Image.init()
        self._image_fnames = sorted(os.path.join(self.img_path,fname) for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        self.files = []
        
        for f in self._image_fnames:
            if not '_mask' in f:
                self.files.append(f)
        
        self.files = sorted(self.files)

    def __len__(self):
        return len(self.files)
    
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_image(self, fn):
        return PIL.Image.open(fn).convert('RGB')
    
    def _get_image(self, idx):
        # imgfn, seg_map, img_id = self.data_reader.get_image(idx)
        
        fname = self.files[idx]
        ext = self._file_ext(fname)
        
        mask = np.array(self._load_image(fname.replace(ext, f'_mask{ext}')).convert('L')) / 255
        mask = cv2.resize(mask,
            (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)

        rgb = np.array(self._load_image(fname)) # uint8
        rgb = cv2.resize(rgb,
            (self.sz, self.sz), interpolation=cv2.INTER_AREA)

        return rgb, fname.split('/')[-1].replace(ext, ''), mask
        
    def __getitem__(self, idx):
        rgb, fname, mask = self._get_image(idx) # modal, uint8 {0, 1}
        rgb = rgb.transpose(2,0,1)

        mask_tensor = torch.from_numpy(mask).to(torch.float32)
        mask_tensor = mask_tensor.unsqueeze(0)
        rgb = torch.from_numpy(rgb.astype(np.float32))
        rgb = (rgb.to(torch.float32) / 127.5 - 1)
        rgb_erased = rgb.clone()
        rgb_erased = rgb_erased * (1 - mask_tensor) # erase rgb
        rgb_erased = rgb_erased.to(torch.float32)
        
        return rgb, rgb_erased, mask_tensor, fname
    
def collate_fn(data):
    """Creates mini-batch tensors from the list of images.
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list
            - image: torch tensor of shape (3, 256, 256).
            
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        
    """

    texture, sketch, fnames = zip(*data)
    
    texture = list(texture)
    sketch = list(sketch)
    fnames = list(fnames)

    return torch.stack(texture, dim=0), torch.stack(sketch, dim=0), fnames

class GarmentDataset3(Dataset):
    def __init__(self,
                 img_path,  # Path to images.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self.sz = resolution
        self.train_path = img_path
        self._type = 'dir'
        self.files = []
        self.idx = 0

        self.real_path = self.train_path + "/image/"
        self.img_path = self.train_path + "/texture/"
        self.sketch_path = self.train_path + "/sketch/"

        self._all_image_fnames = [os.path.relpath(os.path.join(root, fname), start=self.img_path) for root, _dirs, files in
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

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_image(self, fn):
        return PIL.Image.open(fn).convert('RGB')

    def _get_image(self, idx):
        # imgfn, seg_map, img_id = self.data_reader.get_image(idx)

        fname = self.files[idx]
        ext = self._file_ext(fname)
        sketch_fname = os.path.join(self.sketch_path, fname)
        texture_fname = os.path.join(self.img_path, fname)
        real_fname = os.path.join(self.real_path, fname)

        sketch = np.array(self._load_image(sketch_fname)) # uint8
        texture = np.array(self._load_image(texture_fname))
        real = np.array(self._load_image(real_fname))

        return texture, fname.split('/')[-1].replace(ext, ''), sketch, real

    def _put_patch_center(self, texture, img):
        image_size = img.shape[0]
        texture_size = texture.shape[0]
        start_index =(image_size - texture_size) // 2
        end_index = start_index + texture_size
        img[start_index: end_index, start_index: end_index, :] = texture
        return img

    def __getitem__(self, idx):
        texture, fname, sketch, real = self._get_image(idx)  # modal, uint8 {0, 1}
        if texture.shape[0]!=sketch.shape[0]:
            texture_img = np.zeros_like(sketch)
            texture_img = self._put_patch_center(texture, texture_img)
            texture = texture_img

        texture = texture.transpose(2, 0, 1)
        sketch = sketch.transpose(2, 0, 1)
        real = real.transpose(2, 0, 1)

        sketch = torch.from_numpy(sketch.astype(np.float32))
        sketch = (sketch.to(torch.float32) / 127.5 - 1)

        texture = torch.from_numpy(texture.astype(np.float32))
        texture = (texture.to(torch.float32) / 127.5 - 1)

        real= torch.from_numpy(real.astype(np.float32))
        real = (real.to(torch.float32) / 127.5 - 1)

        return texture, sketch, real, fname

class GarmentDataset2(Dataset):
    def __init__(self,
                 img_path,  # Path to images.
                 resolution=None,  # Ensure specific resolution, None = highest available.
                 **super_kwargs,  # Additional arguments for the Dataset base class.
                 ):
        self.sz = resolution
        self.train_path = img_path
        self._type = 'dir'
        self.files = []
        self.idx = 0

        self.img_path = self.train_path + "/texture/"
        self.sketch_path = self.train_path + "/sketch/"

        self._all_image_fnames = [os.path.relpath(os.path.join(root, fname), start=self.img_path) for root, _dirs, files in
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

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_image(self, fn):
        return PIL.Image.open(fn).convert('RGB')

    def _get_image(self, idx):
        # imgfn, seg_map, img_id = self.data_reader.get_image(idx)

        fname = self.files[idx]
        ext = self._file_ext(fname)
        sketch_fname = os.path.join(self.sketch_path, fname)
        texture_fname = os.path.join(self.img_path, fname)

        sketch = np.array(self._load_image(sketch_fname))
        texture = np.array(self._load_image(texture_fname))  # uint8

        return texture, fname.split('/')[-1].replace(ext, ''), sketch

    def _put_patch_center(self, texture, img):
        image_size = img.shape[0]
        texture_size = texture.shape[0]
        start_index =(image_size - texture_size) // 2
        end_index = start_index + texture_size
        img[start_index: end_index, start_index: end_index, :] = texture
        return img

    def __getitem__(self, idx):
        texture, fname, sketch = self._get_image(idx)  # modal, uint8 {0, 1}
        if texture.shape[0]!=sketch.shape[0]:
            texture_img = np.zeros_like(sketch)
            texture_img = self._put_patch_center(texture, texture_img)
            texture = texture_img

        texture = texture.transpose(2, 0, 1)
        sketch = sketch.transpose(2, 0, 1)

        sketch = torch.from_numpy(sketch.astype(np.float32))
        sketch = (sketch.to(torch.float32) / 127.5 - 1)

        texture = torch.from_numpy(texture.astype(np.float32))
        texture = (texture.to(torch.float32) / 127.5 - 1)


        return texture, sketch, fname

def get_loader(img_path, resolution):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    ds = GarmentDataset2(img_path=img_path, resolution=resolution)
    
    data_loader = torch.utils.data.DataLoader(dataset=ds, 
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              collate_fn=collate_fn)
    return data_loader