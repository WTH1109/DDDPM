import os.path
from datetime import time

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import logging


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


def show_difference_heatmap(matrix1, matrix2):
    # 确保两个矩阵的尺寸相同
    if matrix1.shape != matrix2.shape:
        raise ValueError("两个矩阵的尺寸不一致！")

    # 计算两个矩阵之间的绝对差异
    diff = np.abs(matrix1 - matrix2)

    # 归一化差异值到0-255之间
    norm_diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # 使用热图颜色映射将差异值转换为伪彩色图像
    heatmap = cv2.applyColorMap(norm_diff.astype(np.uint8), cv2.COLORMAP_JET)

    # 显示原矩阵和热图
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.title('Matrix 1')
    plt.imshow(matrix1, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Matrix 2')
    plt.imshow(matrix2, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Difference Heatmap')
    plt.imshow(heatmap)
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def all_sim_cau(batch_img_data, batch_target_data, batch_source_data):
    np_batch_img_data = batch_img_data.detach().cpu().numpy()
    np_batch_target_data = batch_target_data.detach().cpu().numpy()
    np_batch_source_data = batch_source_data.detach().cpu().numpy()

    if np_batch_img_data.ndim == 3:
        np_batch_img_data = np.expand_dims(np_batch_img_data, axis=1)
    if np_batch_target_data.ndim == 3:
        np_batch_target_data = np.expand_dims(np_batch_target_data, axis=1)
    if batch_source_data.ndim == 3:
        np_batch_source_data = np.expand_dims(np_batch_source_data, axis=1)

    b, _, _, _ = np_batch_target_data.shape
    ssim_list = []
    dice_list = []
    psnr_list = []
    weight_psnr_list = []
    for i in range(b):
        img_data = np_batch_img_data[i, 0, :, :]
        target_data = np_batch_target_data[i, 0, :, :]
        source_data = np_batch_source_data[i, 0, :, :]
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        target_data = (target_data - np.min(target_data)) / (np.max(target_data) - np.min(target_data))
        source_data = (source_data - np.min(source_data)) / (np.max(source_data) - np.min(source_data))
        match_img_data = hist_match(img_data, target_data,input_type=np.float32)
        show_difference_heatmap(match_img_data, target_data)
        ssim_value, dice_value, psnr_value = sim_cau(match_img_data, target_data, )
        weighted_psnr_value = weighted_psnr(match_img_data, target_data,source_data)
        ssim_list.append(ssim_value)
        dice_list.append(dice_value)
        psnr_list.append(psnr_value)
        weight_psnr_list.append(weighted_psnr_value)
    return ssim_list, dice_list, psnr_list, weight_psnr_list


def avg_sim_cau(batch_img_data, batch_target_data, describe,
                main_path='/mnt/disk10T/home/wengtaohan/Code/latent-diffusion-main', if_save=True):
    np_batch_img_data = batch_img_data.detach().cpu().numpy()
    np_batch_target_data = batch_target_data.detach().cpu().numpy()
    b, _, _, _ = np_batch_target_data.shape
    total_ssim = 0
    total_dice = 0
    total_psnr = 0
    for i in range(b):
        ssim_value, dice_value, psnr_value = sim_cau(np_batch_img_data[i, 0, :, :], np_batch_target_data[i, 0, :, :])
        # plt.imshow(np_batch_img_data[i, 0, :, :], cmap='gray')
        # plt.show()
        # plt.imshow(np_batch_target_data[i, 0, :, :], cmap='gray')
        # plt.show()
        total_ssim += ssim_value
        total_dice += dice_value
        total_psnr += psnr_value
        print('[%s: ssim:%.4f - dice:%.4f - psnr:%.4f' % (describe, ssim_value, dice_value, psnr_value))

    avg_ssim = total_ssim / b
    avg_dice = total_dice / b
    avg_psnr = total_psnr / b

    if if_save:
        log_path = os.path.join(main_path, 'sim_log', 'cta_cond')
        os.makedirs(log_path, exist_ok=True)

        log_filename = os.path.join(log_path, '1.log')
        txt_filename = os.path.join(log_path, '1.txt')
        logging.basicConfig(filename=log_filename, level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info('%s: ssim:%.4f - dice:%.4f - psnr:%.4f' % (describe, avg_ssim, avg_dice, avg_psnr))
        print('avg %s: ssim:%.4f - dice:%.4f - psnr:%.4f' % (describe, avg_ssim, avg_dice, avg_psnr))
        # 打开文本文件以追加模式
        file = open(txt_filename, 'a+')

        # 写入内容
        file.write('\n%s: ssim:%.4f - dice:%.4f - psnr:%.4f' % (describe, avg_ssim, avg_dice, avg_psnr))

        # 关闭文件
        file.close()

    return avg_ssim, avg_dice, avg_psnr


def weighted_psnr(img_data, target_data, source_data):
    img_shape = img_data.shape
    if len(img_shape) == 3:
        _, h, w = img_shape
    else:
        h, w = img_shape
    max_value = img_data.max()

    diff_tar = abs(img_data - target_data)
    diff_ori = abs(source_data - target_data)
    weight = (diff_ori - diff_ori.min()) / (diff_ori.max() - diff_ori.min()) + 1
    normalize_weight = weight / weight.sum() * h * w
    weight_mse = np.sum(diff_tar * diff_tar * normalize_weight) / h / w
    weight_psnr = 10 * np.log10(max_value ** 2 / weight_mse)
    return weight_psnr


def hist_match(img_data, target_data, input_type=np.uint8):
    if input_type != np.uint8:
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        img_data = (img_data * 255).astype(np.uint8)
        target_data = (target_data - np.min(target_data)) / (np.max(target_data) - np.min(target_data))
        target_data = (target_data * 255).astype(np.uint8)

    hist_img, bins_img = np.histogram(img_data.flatten(), bins=256, range=[0, 256])
    hist_target, bins_target = np.histogram(target_data.flatten(), bins=256, range=[0, 256])

    source_cdf = np.cumsum(hist_img) / np.sum(hist_img)
    target_cdf = np.cumsum(hist_target) / np.sum(hist_target)

    mapping = np.interp(source_cdf, target_cdf, range(256)).astype(np.uint8)

    matched_img = mapping[img_data]

    # new_hist_img, new_bins_img = np.histogram(matched_img.flatten(), bins=256, range=[0, 256])

    # plt.bar(bins_img[:-1], hist_img, width=1)
    # plt.title('Pixel Histogram')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.show()
    #
    # plt.bar(bins_target[:-1], hist_target, width=1)
    # plt.title('Pixel Histogram')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.show()
    #
    # plt.bar(bins_target[:-1], new_hist_img, width=1)
    # plt.title('Pixel Histogram')
    # plt.xlabel('Pixel Value')
    # plt.ylabel('Frequency')
    # plt.show()
    #
    #
    #
    # plt.imshow(img_data, cmap='gray')
    # plt.title('ori_img')
    # plt.show()
    #
    # plt.imshow(target_data, cmap='gray')
    # plt.title('target_img')
    # plt.show()
    #
    # plt.imshow(matched_img, cmap='gray')
    # plt.title('matched')
    # plt.show()

    if input_type == np.uint8:
        return matched_img
    else:
        matched_img = (matched_img - np.min(matched_img)) / (np.max(matched_img) - np.min(matched_img))
        return matched_img


if __name__ == '__main__':
    contrast = cv2.imread('/mnt/ssd2/wengtaohan/Code/latent-diffusion-main/imgs/test_contrast.png')[:, :, 0]
    non_contrast = cv2.imread('/mnt/ssd2/wengtaohan/Code/latent-diffusion-main/imgs/test_non_contrast_img_save.png')[:,
                   :, 0]
    dec_contrast = cv2.imread('/mnt/ssd2/wengtaohan/Code/latent-diffusion-main/imgs/test_dec_contrast.png')[:, :, 0]

    dec_contrast_match = hist_match(dec_contrast, contrast)
    dec_contrast = dec_contrast_match

    contrast = (contrast - np.min(contrast)) / (np.max(contrast) - np.min(contrast))
    non_contrast = (non_contrast - np.min(non_contrast)) / (np.max(non_contrast) - np.min(non_contrast))
    dec_contrast = (dec_contrast - np.min(dec_contrast)) / (np.max(dec_contrast) - np.min(dec_contrast))

    contrast = torch.Tensor(contrast)
    non_contrast = torch.Tensor(non_contrast)
    dec_contrast = torch.Tensor(dec_contrast)

    contrast = torch.unsqueeze(contrast, dim=0)
    non_contrast = torch.unsqueeze(non_contrast, dim=0)
    dec_contrast = torch.unsqueeze(dec_contrast, dim=0)

    ssim_list, dice_list, psnr_list, weight_psnr_list = all_sim_cau(dec_contrast, contrast, non_contrast)
    print('done')
