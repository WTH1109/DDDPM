import os
import random

import PIL
import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CTAHeart(Dataset):
    """
    CTA heart dataset
        if construct_method is None, then all sequences will be arranged in order
        if construct_method is 'balance', then the number of samples of each sequence will be balanced
        if construct_method is 'strictly_balance', then the number of samples of
        each sequence will be balanced to same number
    """

    def __init__(self,
                 data_root,
                 yaml_path,
                 size=None,
                 construct_method=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 select_num=10,
                 load_modality_list=None,
                 debug=False
                 ):

        self.debug = debug
        self.data_root = data_root
        self.patient_list = os.listdir(self.data_root)
        self.patient_dic = yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader)['channel']
        self.slice_name_list = []
        self.construction_dataset(method=construct_method)

        self.size = size

        self.low_windows = -100
        self.high_windows = 500

        self.select_num = select_num
        self.patient_num = len(self.patient_dic)

        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        if load_modality_list is None:
            self.load_modality_list = [
                'contrast',
                'non_contrast'
            ]
        else:
            self.load_modality_list = load_modality_list

        self.flip = transforms.RandomHorizontalFlip(p=1)
        self.rotation = transforms.RandomRotation(degrees=90)
        self.flip_p = flip_p

    @staticmethod
    def encode_slice(patient, slice_num):
        return '%s&%d' % (patient, slice_num)

    @staticmethod
    def decode_slice(slice_name):
        patient, slice_num = slice_name.split('&')
        return patient, int(slice_num)

    def construction_dataset(self, method=None):
        # 构建数据集，将结果保存在slice_name_list中
        if method is None:
            for patient, slice_num in self.patient_dic.items():
                for slice_index in range(slice_num):
                    self.slice_name_list.append(self.encode_slice(patient, slice_index))
        elif method == 'balance':
            # 平衡样本之间的不均匀性
            max_slice_num = max(self.patient_dic.values())
            for patient, slice_num in self.patient_dic.items():
                for i in range(max_slice_num // slice_num):
                    for slice_index in range(slice_num):
                        self.slice_name_list.append(self.encode_slice(patient, slice_index))
        elif method == 'strictly_balance':
            # 平衡样本之间的不均匀性
            max_slice_num = max(self.patient_dic.values())
            for patient, slice_num in self.patient_dic.items():
                for i in range(max_slice_num):
                    self.slice_name_list.append(self.encode_slice(patient, i % slice_num))
        return self.slice_name_list

    def __getitem__(self, item):
        tmp_patient, tmp_slice_index = self.decode_slice(self.slice_name_list[item])

        load_cta_dic = {
            load_modality: np.load(os.path.join(self.data_root, tmp_patient,
                                                load_modality + '.npy')) for load_modality in
            self.load_modality_list
        }

        example = {
            'patient_name': tmp_patient,
            'slice': tmp_slice_index
        }

        random_p = random.random()
        for modality in self.load_modality_list:
            cta_slice = load_cta_dic[modality][:, :, tmp_slice_index]
            cta_slice = self.cta_normalize(cta_slice)
            cta_slice = Image.fromarray(cta_slice)
            if self.size is not None:
                cta_slice = cta_slice.resize((self.size, self.size), resample=self.interpolation)
            if random_p < self.flip_p:
                cta_slice = self.flip(cta_slice)
                cta_slice = self.rotation(cta_slice)
            cta_slice = np.array(cta_slice)

            example[modality] = cta_slice

        return example

    def __len__(self):
        if self.debug is False:
            return len(self.slice_name_list)
        else:
            return 8

    def cta_normalize(self, img):
        img[img < -600] = self.low_windows
        img[img > 1500] = self.high_windows
        return 2 * (img - np.min(img)) / (np.max(img) - np.min(img)) - 1


class CTAHeartFULLIMAGE(CTAHeart):
    def __init__(self,
                 data_root,
                 yaml_path,
                 size=None,
                 construct_method=None,
                 interpolation="bicubic",
                 flip_p=0.5,
                 load_modality_list=None,
                 debug=False
                 ):
        super(CTAHeartFULLIMAGE, self).__init__(data_root, yaml_path, size,
                                                construct_method, interpolation, flip_p,
                                                load_modality_list)

    def __len__(self):
        if self.debug is False:
            return len(self.slice_name_list) * len(self.load_modality_list)
        else:
            return 8

    def __getitem__(self, item):
        select_modality = self.load_modality_list[(item // len(self.slice_name_list)) % 2]
        tmp_item = item % len(self.slice_name_list)

        tmp_patient, tmp_slice_index = self.decode_slice(self.slice_name_list[tmp_item])

        load_cta_dic = {
            load_modality: np.load(os.path.join(self.data_root, tmp_patient,
                                                load_modality + '.npy')) for load_modality in
            self.load_modality_list
        }

        example = {
            'patient_name': tmp_patient,
            'slice': tmp_slice_index
        }

        random_p = random.random()

        cta_slice = load_cta_dic[select_modality][:, :, tmp_slice_index]
        cta_slice = self.cta_normalize(cta_slice)
        cta_slice = Image.fromarray(cta_slice)

        if self.size is not None:
            cta_slice = cta_slice.resize((self.size, self.size), resample=self.interpolation)

        if random_p < self.flip_p:
            cta_slice = self.flip(cta_slice)
            cta_slice = self.rotation(cta_slice)

        cta_slice = np.array(cta_slice)

        example['image'] = cta_slice
        example['modality'] = select_modality

        return example


if __name__ == '__main__':
    a = CTAHeartFULLIMAGE('D:/Dataset/CTA(2)/dcm_img_test/third_stage/GE',
                 'D:/Dataset/CTA(2)/dcm_img_test/third_stage/yaml/GE.yaml',
                 construct_method='strictly_balance', flip_p=1, size=256)
    sample = a.__getitem__(10)
    print(a.__len__())
    print('done')
