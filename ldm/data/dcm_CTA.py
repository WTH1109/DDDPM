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

import cv2


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
                 debug=False,
                 normalize_range=None,
                 windows=None,
                 zero_modality_list=None,
                 load_format='npy'
                 ):

        if windows is None:
            windows = [-1000,
                       1000]
        if normalize_range is None:
            self.normalize_range = [-1, 1]
        else:
            self.normalize_range = normalize_range

        self.debug = debug
        self.data_root = data_root
        self.patient_list = os.listdir(self.data_root)
        self.patient_dic = yaml.load(open(yaml_path, 'r'), Loader=yaml.FullLoader)['channel']
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
                'contrast',
                'non_contrast'
            ]
        else:
            self.load_modality_list = load_modality_list

        self.flip = transforms.RandomHorizontalFlip(p=1)
        self.rotation = transforms.RandomRotation(degrees=20, fill=0)
        self.flip_p = flip_p

        self.zero_modality_list = zero_modality_list

        self.load_format = load_format

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

        if self.load_format == 'npy':
            load_cta_dic = {
                load_modality: np.load(os.path.join(self.data_root, tmp_patient,
                                                    load_modality + '.npy')) for load_modality in
                self.load_modality_list
            }
        elif self.load_format == 'nii':
            load_cta_dic = {
                load_modality: nib.load(os.path.join(self.data_root, tmp_patient,
                                                    load_modality + '.nii.gz')).get_fdata() for load_modality in
                self.load_modality_list
            }

        example = {
            'patient_name': tmp_patient,
            'slice': tmp_slice_index
        }

        random_p = random.random()
        seed = np.random.randint(2147483647)
        _, _, slice_num = load_cta_dic[self.load_modality_list[0]].shape

        for modality in self.load_modality_list:
            # tmp_slice_index_increase = tmp_slice_index
            # while 1:
            #     cta_slice = load_cta_dic[modality][:, :, tmp_slice_index_increase]
            #     cta_slice = self.cta_normalize(cta_slice)
            #     if cta_slice is not None:
            #         tmp_slice_index = tmp_slice_index_increase
            #         break
            #     tmp_slice_index_increase += 1
            #     tmp_slice_index_increase = tmp_slice_index_increase % slice_num

            cta_slice = load_cta_dic[modality][:, :, tmp_slice_index]

            cta_slice = Image.fromarray(cta_slice)
            if self.size is not None:
                cta_slice = cta_slice.resize((self.size, self.size), resample=self.interpolation)
            if random_p < self.flip_p:
                cta_slice = self.flip(cta_slice)
                random.seed(seed)
                torch.manual_seed(seed)
                cta_slice = self.rotation(cta_slice)
            cta_slice = np.array(cta_slice)

            cta_slice = self.cta_normalize(cta_slice)

            example[modality] = cta_slice

        if self.zero_modality_list is not None:
            for zero_modality in self.zero_modality_list:
                example[zero_modality] = np.zeros(example[zero_modality].shape)

        return example

    def __len__(self):
        if self.debug is False:
            return len(self.slice_name_list)
        else:
            return 8

    def cta_normalize(self, img):
        img[img < self.low_windows] = self.low_windows
        img[img > self.high_windows] = self.high_windows
        if np.max(img) - np.min(img) != 0:
            return ((self.normalize_range[1] - self.normalize_range[0]) * (img - np.min(img)) / (np.max(img) - np.min(img))
                    + self.normalize_range[0])
        else:
            return None


class CTAHeartDeformable(CTAHeart):
    def __init__(self,
                 data_root,
                 yaml_path,
                 deformable_num=5,
                 wrap_name='wrap',
                 zero_prob=0.3,
                 *args, **kwargs
                 ):
        super(CTAHeartDeformable, self).__init__(data_root, yaml_path, *args, **kwargs)
        self.deformable_num = deformable_num
        self.wrap_name = wrap_name
        self.zero_prob = zero_prob


    def __getitem__(self, item):
        tmp_patient, tmp_slice_index = self.decode_slice(self.slice_name_list[item])

        if self.load_format == 'npy':
            load_cta_dic = {
                load_modality: np.load(os.path.join(self.data_root, tmp_patient,
                                                    load_modality + '.npy')) for load_modality in
                self.load_modality_list
            }
        elif self.load_format == 'nii':
            load_cta_dic = {
                load_modality: nib.load(os.path.join(self.data_root, tmp_patient,
                                                    load_modality + '.nii.gz')).get_fdata() for load_modality in
                self.load_modality_list
            }





        random_p = random.random()
        seed = np.random.randint(2147483647)

        _, _, slice_num = load_cta_dic[self.load_modality_list[0]].shape
        wrap_index = [random.randint(max(0, tmp_slice_index - 15), min(slice_num - 1, tmp_slice_index + 15)) for _ in range(self.deformable_num-1)]
        wrap_index.insert(0, tmp_slice_index)

        example = {
            'patient_name': tmp_patient,
            'slice': tmp_slice_index,
            'total_slice': slice_num
        }


        # wrap_data = np.resize(wrap_data, (b, c, self.size, self.size))


        for modality in self.load_modality_list:
            # tmp_slice_index_increase = tmp_slice_index
            # while 1:
            #     cta_slice = load_cta_dic[modality][:, :, tmp_slice_index_increase]
            #     cta_slice = self.cta_normalize(cta_slice)
            #     if cta_slice is not None:
            #         tmp_slice_index = tmp_slice_index_increase
            #         break
            #     tmp_slice_index_increase += 1
            #     tmp_slice_index_increase = tmp_slice_index_increase % slice_num

            cta_slice = load_cta_dic[modality][:, :, tmp_slice_index]


            cta_slice = Image.fromarray(cta_slice)
            if self.size is not None:
                cta_slice = cta_slice.resize((self.size, self.size), resample=self.interpolation)
            if random_p < self.flip_p:
                cta_slice = self.flip(cta_slice)
                random.seed(seed)
                torch.manual_seed(seed)
                cta_slice = self.rotation(cta_slice)
            cta_slice = np.array(cta_slice)

            cta_slice = self.cta_normalize(cta_slice)

            example[modality] = cta_slice

        if self.wrap_name == 'wrap':
            wrap_data = np.load(os.path.join(self.data_root, tmp_patient,
                                             self.wrap_name + '.npy'))
            b, c, h, w = wrap_data.shape
            wrap_np = np.zeros((self.deformable_num, 2, self.size, self.size))
            for i, index_item in enumerate(wrap_index):
                # wrap_slice = wrap_data[index_item, :, :, :]
                random_p = random.random()
                if random_p < self.zero_prob:
                    continue
                wrap_slice_x = wrap_data[index_item, 0, :, :]
                wrap_slice_y = wrap_data[index_item, 1, :, :]
                wrap_slice_x = cv2.resize(wrap_slice_x, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
                wrap_slice_y = cv2.resize(wrap_slice_y, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
                wrap_slice_x = wrap_slice_x * self.size / h
                wrap_slice_y = wrap_slice_y * self.size / h
                wrap_np[i, 0, :, :] = wrap_slice_x
                wrap_np[i, 1, :, :] = wrap_slice_y
                wrap_np = wrap_np.astype(np.float32)
        elif self.wrap_name == 'slice':
            wrap_np = []
            for i, index_item in enumerate(wrap_index):
                other_slice = {}
                # wrap_slice = wrap_data[index_item, :, :, :]
                random_p = random.random()

                for modality in self.load_modality_list:
                    if random_p < self.zero_prob:
                        other_slice[modality] = example[self.load_modality_list[0]]
                        continue

                    # tmp_index_item = index_item
                    # while 1:
                    #     cta_slice = load_cta_dic[modality][:, :, tmp_index_item]
                    #     cta_slice = self.cta_normalize(cta_slice)
                    #     if cta_slice is not None:
                    #         index_item = tmp_index_item
                    #         break
                    #     tmp_index_item += 1
                    #     tmp_index_item = tmp_index_item % slice_num
                    cta_slice = load_cta_dic[modality][:, :, index_item]

                    cta_slice = Image.fromarray(cta_slice)
                    if self.size is not None:
                        cta_slice = cta_slice.resize((self.size, self.size), resample=self.interpolation)
                    if random_p < self.flip_p:
                        cta_slice = self.flip(cta_slice)
                        random.seed(seed)
                        torch.manual_seed(seed)
                        cta_slice = self.rotation(cta_slice)
                    cta_slice = np.array(cta_slice)

                    cta_slice = self.cta_normalize(cta_slice)

                    other_slice[modality] = cta_slice
                wrap_np.append(other_slice)
        else:
            wrap_np = 0
        if self.zero_modality_list is not None:
            for zero_modality in self.zero_modality_list:
                example[zero_modality] = np.zeros(example[zero_modality].shape)

        example[self.wrap_name] = wrap_np

        return example


class CTAHeartFULLIMAGE(CTAHeart):
    def __init__(self,
                 data_root,
                 yaml_path,
                 *args, **kwargs
                 ):
        super(CTAHeartFULLIMAGE, self).__init__(data_root, yaml_path, *args, **kwargs)

    def __len__(self):
        if self.debug is False:
            return len(self.slice_name_list) * len(self.load_modality_list)
        else:
            return 8

    def __getitem__(self, item):
        select_modality = self.load_modality_list[(item // len(self.slice_name_list)) % 2]
        tmp_item = item % len(self.slice_name_list)

        tmp_patient, tmp_slice_index = self.decode_slice(self.slice_name_list[tmp_item])


        if self.load_format == 'npy':
            load_cta_dic = {
                load_modality: np.load(os.path.join(self.data_root, tmp_patient,
                                                    load_modality + '.npy')) for load_modality in
                self.load_modality_list
            }
        elif self.load_format == 'nii':
            load_cta_dic = {
                load_modality: nib.load(os.path.join(self.data_root, tmp_patient,
                                                    load_modality + '.nii.gz')).get_fdata() for load_modality in
                self.load_modality_list
            }

        _, _, slice_num = load_cta_dic[self.load_modality_list[0]].shape

        example = {
            'patient_name': tmp_patient,
            'slice': tmp_slice_index,
            'total_slice': slice_num
        }


        random_p = random.random()

        # tmp_slice_index_increase = tmp_slice_index
        # while 1:
        #     cta_slice = load_cta_dic[select_modality][:, :, tmp_slice_index_increase]
        #     cta_slice = self.cta_normalize(cta_slice)
        #     if cta_slice is not None:
        #         tmp_slice_index = tmp_slice_index_increase
        #         break
        #     tmp_slice_index_increase += 1
        #     tmp_slice_index_increase = tmp_slice_index_increase % slice_num

        cta_slice = load_cta_dic[select_modality][:, :, tmp_slice_index]

        cta_slice = Image.fromarray(cta_slice)

        if self.size is not None:
            cta_slice = cta_slice.resize((self.size, self.size), resample=self.interpolation)

        if random_p < self.flip_p:
            cta_slice = self.flip(cta_slice)
            cta_slice = self.rotation(cta_slice)

        cta_slice = np.array(cta_slice)

        cta_slice = self.cta_normalize(cta_slice)

        example['image'] = cta_slice
        example['modality'] = select_modality

        return example



if __name__ == '__main__':
    data_root = '/mnt/ssd2/wengtaohan/Data/CTA/Siemens/voxel_register'
    # data_root = '/mnt/disk10T/home/wengtaohan/database/CTA/dcm_img/dataset'
    a = CTAHeartDeformable(os.path.join(data_root, 'train'), os.path.join(data_root, 'yaml/train.yaml'),deformable_num=5,wrap_name='slice',
                 construct_method='strictly_balance', flip_p=0, size=256, normalize_range=[0, 1],
                 zero_modality_list=None)
    sample = a.__getitem__(10)
    plt.imshow(sample['contrast'], cmap='gray')
    plt.show()
    plt.imshow(sample['non_contrast'], cmap='gray')
    plt.show()
    plt.imshow(sample['wrap'][0,0,:,:], cmap='gray')
    plt.show()
    print(a.__len__())
    print('done')
