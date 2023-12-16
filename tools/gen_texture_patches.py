import os
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def get_texture(groundtruth, mask):
    deafult_encode_size = 64
    max_encode_size = 96
    encode_size = random.randint(deafult_encode_size, max_encode_size)

    image_size = groundtruth.shape[0]

    max_loop = 1000
    for i in range(max_loop):
        x = random.randint(0, image_size - encode_size - 1)
        y = random.randint(0, image_size - encode_size - 1)
        mask_crop = mask[x: x + encode_size, y:y + encode_size,:]
        if 0.98 <= np.mean(mask_crop)/255 <= 1:
            patch = groundtruth[x: x + encode_size, y:y + encode_size, :]
            return patch
        encode_size = random.randint(deafult_encode_size, max_encode_size)

    # avoid die
    # center crop
    clip_start_index = (image_size - deafult_encode_size) // 2
    clip_end_index = clip_start_index + deafult_encode_size
    patch = groundtruth[clip_start_index: clip_end_index, clip_start_index: clip_end_index, :]
    return patch


def put_patch_center(texture, img):
    image_size = img.shape[0]
    texture_size = texture.shape[0]
    start_index = (image_size - texture_size) // 2
    end_index = start_index + texture_size
    img[start_index: end_index, start_index: end_index, :] = texture
    return img


def gen_texture_patches(img_data, mask_data, save_data):
    img_base = Path(img_data)
    # mask_base = Path(mask_data)
    save_base = Path(save_data)
    if not os.path.exists(save_base):
        os.makedirs(save_base)

    print('making patch...')
    img_len = len(list(img_base.iterdir()))
    for img_f in tqdm(img_base.iterdir(), total=img_len):
        img = cv2.imread(str(img_f), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_data + "/" + img_f.name), cv2.IMREAD_COLOR)

        mask[mask > 235] = 255
        mask[mask <= 235] = 0
        mask = np.expand_dims(mask[:, :, 0], axis=2)

        patch = get_texture(img, mask)
        cv2.imwrite(str(str(save_data) + '/' + img_f.name), patch)


img_data = "your_path_for_val/image"
mask_data = "your_path_for_val/mask"
save_data = "your_path_for_val/texture"
gen_texture_patches(img_data, mask_data, save_data)
