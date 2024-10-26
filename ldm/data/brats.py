import os
import random

import PIL
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import nibabel as nib


class Brats(Dataset):
    """
    Brats dataset
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
                 debug=False,
                 normalize_range=None,
                 windows=None,
                 zero_modality_list=None
                 ):

        if windows is None:
            windows = [-1000,
                       1000]
        if normalize_range is None:
            self.normalize_range = [-1, 1]
        else:
            self.normalize_range = normalize_range

        self.slice_num = 60
        self.begin_slice = 50

        self.debug = debug
        self.data_root = data_root
        self.patient_list = os.listdir(self.data_root)
        self.patient_list = [item for item in self.patient_list if os.path.isdir(os.path.join(self.data_root, item))]
        self.patient_dic = {patient: self.slice_num for patient in self.patient_list}
        self.slice_name_list = []
        self.construction_dataset(method=construct_method)

        self.size = size



        # [-100,500]
        self.low_windows = windows[0]
        self.high_windows = windows[1]

        self.select_num = select_num
        self.patient_num = len(self.patient_dic)

        self.interpolation = {"bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]

        if load_modality_list is None:
            self.load_modality_list = [
                't1',
                't2',
                't1ce',
                'flair'
            ]
        else:
            self.load_modality_list = load_modality_list

        self.flip = transforms.RandomHorizontalFlip(p=1)
        self.rotation = transforms.RandomRotation(degrees=20, fill=0)
        self.flip_p = flip_p

        self.zero_modality_list = zero_modality_list

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

        tmp_slice_index = tmp_slice_index + self.begin_slice

        load_mri_dic = {
            load_modality: nib.load(os.path.join(self.data_root, tmp_patient,
                                                 tmp_patient + '_' + load_modality + '.nii')).get_fdata() for
            load_modality in self.load_modality_list
        }

        example = {
            'patient_name': tmp_patient,
            'slice': tmp_slice_index
        }

        random_p = random.random()
        seed = np.random.randint(2147483647)
        for modality in self.load_modality_list:
            mri_slice = load_mri_dic[modality][:, :, tmp_slice_index]
            mri_slice = self.mri_normalize(mri_slice)
            mri_slice = Image.fromarray(mri_slice)
            if self.size is not None:
                mri_slice = mri_slice.resize((self.size, self.size), resample=self.interpolation)
            if random_p < self.flip_p:
                mri_slice = self.flip(mri_slice)
                random.seed(seed)
                torch.manual_seed(seed)
                mri_slice = self.rotation(mri_slice)
            mri_slice = np.array(mri_slice)

            example[modality] = mri_slice

        if self.zero_modality_list is not None:
            for zero_modality in self.zero_modality_list:
                example[zero_modality] = np.zeros(example[zero_modality].shape)

        return example

    def __len__(self):
        if self.debug is False:
            return len(self.slice_name_list)
        else:
            return 8

    def mri_normalize(self, img):
        img[img < self.low_windows] = self.low_windows
        img[img > self.high_windows] = self.high_windows
        return ((self.normalize_range[1] - self.normalize_range[0]) * (img - np.min(img)) / (np.max(img) - np.min(img))
                + self.normalize_range[0])


class BratsFULLIMAGE(Brats):
    def __init__(self,
                 data_root,
                 yaml_path,
                 *args, **kwargs
                 ):
        super(BratsFULLIMAGE, self).__init__(data_root, yaml_path, *args, **kwargs)

    def __len__(self):
        if self.debug is False:
            return len(self.slice_name_list) * len(self.load_modality_list)
        else:
            return 8

    def __getitem__(self, item):
        select_modality = self.load_modality_list[(item // len(self.slice_name_list)) % 4]
        tmp_item = item % len(self.slice_name_list)

        tmp_patient, tmp_slice_index = self.decode_slice(self.slice_name_list[tmp_item])

        tmp_slice_index = tmp_slice_index + self.begin_slice

        load_mri_dic = {
            load_modality: nib.load(os.path.join(self.data_root, tmp_patient,
                                                 tmp_patient + '_' + load_modality + '.nii')).get_fdata() for
            load_modality in self.load_modality_list
        }

        example = {
            'patient_name': tmp_patient,
            'slice': tmp_slice_index
        }

        random_p = random.random()

        mri_slice = load_mri_dic[select_modality][:, :, tmp_slice_index]
        mri_slice = self.mri_normalize(mri_slice)
        mri_slice = Image.fromarray(mri_slice)

        if self.size is not None:
            mri_slice = mri_slice.resize((self.size, self.size), resample=self.interpolation)

        if random_p < self.flip_p:
            mri_slice = self.flip(mri_slice)
            mri_slice = self.rotation(mri_slice)

        mri_slice = np.array(mri_slice)

        example['image'] = mri_slice
        example['modality'] = select_modality

        return example


if __name__ == '__main__':
    data_root = '/mnt/ssd2/wengtaohan/Data/BraTs2020/data/BraTS2020_TrainingData'
    # data_root = '/mnt/disk10T/home/wengtaohan/database/CTA/dcm_img/dataset'
    a = BratsFULLIMAGE(os.path.join(data_root, 'MICCAI_BraTS2020_TrainingData'), os.path.join(data_root, 'yaml/GE_train.yaml'),
              construct_method='strictly_balance', flip_p=1, size=256, normalize_range=[0, 1])
    sample = a.__getitem__(20)
    plt.imshow(sample['image'], cmap='gray')
    plt.show()
    # plt.imshow(sample['t2'], cmap='gray')
    # plt.show()
    print(a.__len__())
    print('done')
