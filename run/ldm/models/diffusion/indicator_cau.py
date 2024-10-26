import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def dice_coefficient(image1, image2):
    image1 = image1 > 0.5
    image2 = image2 > 0.5
    intersection = np.logical_and(image1, image2).sum()
    union = image1.sum() + image2.sum()
    dice = (2 * intersection) / union
    return dice


def sim_cau(img_data, target_data):
    img_data_normalize = (img_data - np.min(img_data)) / (
            np.max(img_data) - np.min(img_data))
    target_data_normalize = (target_data - np.min(target_data)) / (
            np.max(target_data) - np.min(target_data))

    ssim_value = ssim(img_data_normalize, target_data_normalize, multichannel=True, data_range=1)
    dice_value = dice_coefficient(img_data_normalize, target_data_normalize)
    psnr_value = psnr(img_data_normalize, target_data_normalize, data_range=1)
    return ssim_value, dice_value, psnr_value
