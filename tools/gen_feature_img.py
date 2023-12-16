import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import torch

def display_feature(feature, from_model):
    assert feature.dim()==4
    feature_size = feature.size(-1)
    name = from_model + str(feature_size)
    # feature_img = feature[0, 1, :, :].detach().cpu()
    #- feature_img = feature_img.unsqueeze(dim=0).repeat(3,1,1)
    #- save_image_grid(feature_img, name)
    # save_feature_grid(feature_img, name)
    feature = feature[0,0:30,:,:]
    for i in range(feature.size(0)):
        feature_img = feature[i, :, :].detach().cpu()
        name = from_model + str(feature_size)+"_"+str(i)
        save_feature_grid(feature_img, name)

def save_image_grid(tensor, name):
    #lo, hi = [-1, 1]

    tensor = np.asarray(tensor.cpu(), dtype=np.float32).transpose(1, 2, 0)
    lo = ten_min = np.min(tensor)
    hi = ten_max = np.max(tensor)
    # to -1,1
    #ten_min = np.min(tensor)
    #ten_max = np.max(tensor)
    #tensor = (tensor - ten_min) / (ten_max - ten_min)
    #tensor = tensor * (hi-lo) - (hi-lo)/2
    # to 0-255
    tensor = (tensor - lo) * (255 / (hi - lo))
    tensor = np.rint(tensor).clip(0, 255).astype(np.uint8)

    plt.imsave('./feature/' + name + '.png', tensor )# / 255

def save_feature_grid(tensor, name):
    plt.axis('off')
    plt.imshow(tensor)
    plt.savefig('./feature/' + name + '.png')

#------------------------------------------------------------
def get_amplitude(complex_img):
    real = torch.pow(complex_img.real, 2.0)
    imaginary = torch.pow(complex_img.imag, 2.0)
    amplitude = torch.sqrt(real + imaginary)
    amplitude_log = torch.log(amplitude + 1)
    return amplitude_log

def display_frequency_feature(feature, from_model):
    assert feature.dim()==4

    feature_amp = get_amplitude(feature)

    feature_size = feature_amp.size(-1)
    name = from_model + str(feature_size)
    # feature_img = feature_amp[0, 1, :, :].detach().cpu()
    #- feature_img = feature_img.unsqueeze(dim=0).repeat(3,1,1)
    #- save_image_grid(feature_img, name)
    # save_feature_grid(feature_img, name)

    feature = feature_amp[0,0:30,:,:]
    for i in range(feature.size(0)):
        feature_img = feature[i, :, :].detach().cpu()
        name = from_model + str(feature_size)+"_"+str(i)
        save_feature_grid(feature_img, name)
