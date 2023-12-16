import torch
import torch.nn as nn
import numpy as np
import torchvision.models as tvmodels

class Frequency_PercLoss(nn.Module):
    def __init__(self, vgg_module, select_layers=None, frequency_mode="RGB", what_to_cul="amplitude"):
        super(Frequency_PercLoss, self).__init__()
        self.vgg_module = vgg_module
        self.select_layers = select_layers
        if self.select_layers is None:
            self.select_layers =['1',
                                  '3',
                                  '6',
                                  '8',
                                  '11',
                                  '13',
                                  '15',
                                  '18',
                                  '20',
                                  '22',
                                  '25',
                                  '27',
                                  '29']

        self.criterion = nn.L1Loss()  # nn.MSELoss()
        self.frequency_mode = frequency_mode
        self.what_to_cul = what_to_cul

    def __call__(self, input, target):
        self.vgg_module = self.vgg_module.to(input.device)
        # loss
        loss_perc = 0.0

        # inputs are pics
        input = self.normalize(input, o_min=-1, o_max=1, t_min=0, t_max=1)
        target = self.normalize(target, o_min=-1, o_max=1, t_min=0, t_max=1)

        if target.dim() == 4:  # RGB
            b, c, h, w = target.size()
        elif target.dim() == 3:  # gray
            b, h, w = target.size()

        # to fft
        input_fft = torch.fft.fft2(input)
        target_fft = torch.fft.fft2(target)

        # downsample fft
        if input_fft.size() != target_fft.size():
            input_fft = self.adaptive_avg_pool2d_complex(input_fft, (h, w))

        if self.what_to_cul == "amplitude":
            input_fft_amplitude = self.get_amplitude(input_fft)
            target_fft_amplitude = self.get_amplitude(target_fft)
            feature_maps_large = self.vgg_module.forward(input_fft_amplitude, self.select_layers)
            feature_maps_small = self.vgg_module.forward(target_fft_amplitude, self.select_layers)
            for input_feature, target_feature in zip(feature_maps_large, feature_maps_small):
                loss_perc = loss_perc + self.criterion(input_feature, target_feature.detach())
            loss_perc = loss_perc / len(self.select_layers)

        elif self.what_to_cul == "frequency":
            feature_maps_large_real = self.vgg_module.forward(input_fft.real, self.select_layers)
            feature_maps_small_real = self.vgg_module.forward(target_fft.real, self.select_layers)
            feature_maps_large_imag = self.vgg_module.forward(input_fft.imag, self.select_layers)
            feature_maps_small_imag = self.vgg_module.forward(target_fft.imag, self.select_layers)
            real_loss = 0.0
            imag_loss = 0.0
            for input_feature, target_feature in zip(feature_maps_large_real, feature_maps_small_real):
                real_loss = real_loss + self.criterion(input_feature, target_feature.detach())
            real_loss = real_loss / len(self.select_layers)
            for input_feature, target_feature in zip(feature_maps_large_imag, feature_maps_small_imag):
                imag_loss = imag_loss + self.criterion(input_feature, target_feature.detach())
            imag_loss = imag_loss / len(self.select_layers)
            loss_perc = real_loss + imag_loss

        return loss_perc

    def RGB_normalize_2image(self, image1, image2):
        """histmatch"""
        # rgb_image1 = image1.copy()
        # rgb_image2 = image2.copy()
        rgb_image1 = np.asarray(image1.cpu().float().numpy())
        rgb_image2 = np.asarray(image2.cpu().float().numpy())
        """histmatch"""
        for i in range(rgb_image1.shape[0]):
            rgb_image1[i, :, :, :] = self.histMatch(rgb_image1[i, :, :, :], rgb_image2[i, :, :, :])  # hist match
        """RGB process"""
        rgb_image1 = torch.from_numpy(rgb_image1).cuda()
        rgb_image2 = torch.from_numpy(rgb_image2).cuda()
        # rgb_image1 = self.rgb2gray_tensor(rgb_image1)
        # rgb_image2 = self.rgb2gray_tensor(rgb_image2)
        """-----------"""
        return rgb_image1, rgb_image2

    def histMatch(self, imsrc, imtint):
        nbr_bins = 255
        # imsrc = to_img(imsrc)
        # imtint = to_img(imtint)
        imres = imsrc.copy()
        for d in range(imsrc.shape[0]):
            # calculate histogram of each image
            imhist, bins = np.histogram(imsrc[d, :, :].flatten(), nbr_bins, density=True)
            tinthist, bins = np.histogram(imtint[d, :, :].flatten(), nbr_bins, density=True)
            # cumulative distribution function of reference image
            cdfsrc = imhist.cumsum()
            cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8)  # normalize
            # cumulative distribution function of target image
            cdftint = tinthist.cumsum()
            cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8)  # normalize
            # use linear interpolation of cdf to find new pixel values
            im2 = np.interp(imsrc[d, :, :].flatten(), bins[:-1], cdfsrc)
            im3 = np.interp(im2, cdftint, bins[:-1])
            imres[d, :, :] = im3.reshape((imsrc.shape[1], imsrc.shape[2]))
        # imres = to_torch(imres)
        return imres

    def adaptive_avg_pool2d_complex(self, fft_image, size):
        real = fft_image.real
        imag = fft_image.imag
        down_real = torch.nn.functional.adaptive_avg_pool2d(real, size)
        down_imag = torch.nn.functional.adaptive_avg_pool2d(imag, size)
        return down_real + (down_imag * 1j)

    def get_amplitude(self, complex_img):
        real = torch.pow(complex_img.real, 2.0)
        imaginary = torch.pow(complex_img.imag, 2.0)
        amplitude = torch.sqrt(real + imaginary)
        amplitude_log = torch.log(amplitude + 1)
        return amplitude_log

    def normalize(self, input, o_min, o_max, t_min, t_max):
        output = t_min + ((input - o_min) / (o_max - o_min)) * (t_max - t_min)
        return output

    def rgb2gray_tensor(self, img):
        if img.dim() == 4:
            gray_box = None
            for i in range(img.size(0)):
                gray = img[i, 0, :, :] * 0.3 + img[i, 1, :, :] * 0.59 + img[i, 2, :, :] * 0.11
                if i == 0:
                    gray_box = torch.unsqueeze(gray, 0)
                else:
                    torch.cat([gray_box, gray], 0)

        elif img.dim() == 3:
            gray = img[:, 0, :, :] * 0.3 + img[:, 1, :, :] * 0.59 + img[:, 2, :, :] * 0.11
            gray_box = torch.unsqueeze(gray, 0)

        return gray_box

class Frequency_AutocorrLoss(nn.Module):
    def __init__(self, vgg_module, select_layers=None):
        super(Frequency_AutocorrLoss, self).__init__()
        self.vgg_module = vgg_module
        self.select_layers = select_layers
        if self.select_layers is None:
            self.select_layers =['1',
                                  '3',
                                  '6',
                                  '8',
                                  '11',
                                  '13',
                                  '15',
                                  '18',
                                  '20',
                                  '22',
                                  '25',
                                  '27',
                                  '29']

        self.criterion = nn.MSELoss()

    def __call__(self, input, target):
        self.vgg_module = self.vgg_module.to(input.device)

        # inputs are pics
        # input = self.normalize(input, o_min=-1, o_max=1, t_min=0, t_max=1)
        # target = self.normalize(target, o_min=-1, o_max=1, t_min=0, t_max=1)

        """
        Computation of the autocorrelation of the filters
        """

        length_used_layers_int = len(self.select_layers)
        length_used_layers = float(length_used_layers_int)
        weight_help_convergence = (10 ** 9)
        total_texture_loss = 0.

        _, N, h_a, w_a = target.shape

        feature_maps_input = self.vgg_module.forward(input, self.select_layers)
        feature_maps_target = self.vgg_module.forward(target, self.select_layers)
        for x, a in zip(feature_maps_input, feature_maps_target):
            M = x.shape[-2]*x.shape[-1]
            N = x.shape[1]
            F_x = torch.fft.fft2(x)
            R_x = torch.real(torch.mul(F_x, torch.conj(F_x)))  # Module de la transformee de Fourrier : produit terme a terme
            R_x /= M ** 2  # Normalisation du module de la TF
            F_a = torch.fft.fft2(a)
            R_a = torch.real(torch.mul(F_a, torch.conj(F_a)))  # Module de la transformee de Fourrier
            R_a /= M ** 2
            texture_loss = self.criterion(torch.subtract(R_x, R_a))
            texture_loss *= weight_help_convergence / (2. * (N ** 2) * length_used_layers)
            total_texture_loss += texture_loss
            total_texture_loss = total_texture_loss.type(torch.float32)

        return total_texture_loss

# features from vgg16
class VGG_Features16(nn.Module):
    def __init__(self):
        super(VGG_Features16, self).__init__()
        self.vgg16 = tvmodels.vgg16(pretrained=True).features.eval()
        for param in self.vgg16.parameters():
            param.requires_grad = False

    def forward(self, input, select_layers=['13', '22']):
        layer_name_mapping = {1: 'relu_1_1',
                              3: 'relu_1_2',
                              6: 'relu_2_1',
                              8: 'relu_2_2',
                              11: 'relu_3_1',
                              13: 'relu_3_2',
                              15: 'relu_3_3',
                              18: 'relu_4_1',
                              20: 'relu_4_2',
                              22: 'relu_4_3',
                              25: 'relu_5_1',
                              27: 'relu_5_2',
                              29: 'relu_5_3',}
        features = []
        for name, layer in self.vgg16._modules.items():
            input = layer(input)
            if name in select_layers:
                features.append(input)
        return features


def define_VGGF16():
    netVGGF16 = VGG_Features16()
    """
    if torch.cuda.is_available():
        netVGGF16.cuda()
    """
    return netVGGF16