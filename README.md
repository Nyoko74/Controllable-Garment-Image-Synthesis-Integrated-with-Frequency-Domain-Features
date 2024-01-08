# Controllable Garment Image Synthesis Integrated with Frequency Domain Features
The repository for Controllable Garment Image Synthesis Integrated with Frequency Domain Features (PG2023).

[paper](https://onlinelibrary.wiley.com/doi/10.1111/cgf.14938)

## Requirements

Install Conda environment:
```
conda env create -f environment.yml
conda activate CGISgan
```

Download ade20k pretrain modal for High Receptive Field Perceptual Loss:
```
mkdir -p ade20k/ade20k-resnet50dilated-ppm_deepsup/
wget -P ade20k/ade20k-resnet50dilated-ppm_deepsup/ http://sceneparsing.csail.mit.edu/model/pytorch/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth
```
<!--
## Pretrain Modal
You can obtain pretrain model by clicking [here](url).
The pretrain model is trained on $256 \times 256$ garment images.
-->
 
## Dataset
Our dataset is built based on the dataset of [FashionGAN](https://github.com/Cuiyirui/FashionGAN) ([https://drive.google.com/drive/folders/1DACqCXlJRQxRysO6RVNO7vOoR8YzrjTQ](https://drive.google.com/drive/folders/1DACqCXlJRQxRysO6RVNO7vOoR8YzrjTQ)).

We have re-extracted the masks and contour lines as sketch from the ground truth in this dataset. 
Regarding the method of mask extraction, you can refer to works such as [BAS-net](https://github.com/xuebinqin/BASNet) and [Segment Anything](https://github.com/facebookresearch/segment-anything).

If you need to prepare your own dataset, please organize it in the following way.

    .\dataset
        └ \train
            └ \image
                └ 1.jpg
                └ 2.jpg
                └ ...
            └ \sketch
                └ 1.jpg
                └ 2.jpg
                └ ...
            └ \mask
                └ 1.jpg
                └ 2.jpg
                └ ...
        └ \val
            └ \image
                └ 1.jpg
                └ 2.jpg
                └ ...
            └ \sketch
                └ 1.jpg
                └ 2.jpg
                └ ...
            └ \mask
                └ 1.jpg
                └ 2.jpg
                └ ...
            └ \texture
                └ 1.jpg
                └ 2.jpg
                └ ...

### Description to The Floder `\train`  
Regarding the sub folders and data descriptions in the train folder:

    Image: Garment ground truth
    Sketch: Garment sketch
    Mask: Garment Mask

For a data unit, the image, its corresponding sketch, and the corresponding mask are named with the same file name, for example:

    \image\1.jpg
    \sketch\1.jpg
    \mask\1.jpg

### Description to The Floder `\val`  
For the `\texture` data in fload `\val`, you can use the script `.\tools\gen_texture_patches.py` to extract it. 
You need to make some edit to the script to generate your texture: (in `line 55-67`)

    img_data = "your_path_for_val\image"
    mask_data = "your_path_for_val\mask"
    save_data = "your_path_for_val\texture"


## Train

    python train.py --outdir [your_output_dir] --img_data [your_train_dataset_dir] --eval_img_data [your_eval_dataset_dir] --gpus 1 --kimg 25000 --gamma 10 --aug noaug --metrics True --batch 8 --snap 1

you can change `--batch` to better utilize your GPU.

## Evaluation

To evaluate your model, use the following command:

    python evaluate.py --img_data [your_eval_dataset_dir] --network [your_model_file] --num_gpus 1

You can test your model with `demo.py`:

    python demo.py --img_data [your_test_dataset_dir] --network [your_model_file] --resolution 256

The format of test set requirements is consistent with the evaluation set.

## Citation

If this work is helpful to you, please cite:

    @inproceedings{liang2023controllable,
    title={Controllable Garment Image Synthesis Integrated with Frequency Domain Features},
    author={Liang, Xinru and Mo, Haoran and Gao, Chengying},
    booktitle={Computer Graphics Forum},
    pages={e14938},
    year={2023},
    organization={Wiley Online Library}
    }

## Acknowledgement

Code is heavily based on [FcF-Inpainting](https://github.com/SHI-Labs/FcF-Inpainting).
