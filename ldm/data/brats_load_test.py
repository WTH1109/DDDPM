import nibabel as nib

data_root = '/mnt/ssd2/wengtaohan/Data/BraTs2020/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/BraTS20_Training_002/BraTS20_Training_002_t1.nii'
data = nib.load(data_root)
print('done')