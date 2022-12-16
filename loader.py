import os
import torch
import numpy as np
from torchvision import transforms
from glob import glob
from torch.utils.data import Dataset, DataLoader


class ct_dataset(Dataset):
    def __init__(self, mode, load_mode, train_path, saved_path, test_patient, 
                patch_n=None, patch_size=None, transform=None, norm = False, patch_training = False):
        assert mode in ['train', 'test'], "mode is 'train' or 'test'"
        assert load_mode in [0,1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(saved_path, '*_input*.npy')))   # glob遍历文件夹下所有文件或文件夹；sorted对所有可迭代的对象进行排序操作
        target_path = sorted(glob(os.path.join(saved_path, '*_target*.npy')))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size
        self.patch_training = patch_training
        self.transform = transform
        self.norm = norm
        self.mode = mode

        if mode == 'train':
            # ugly refactoring
            input_path = sorted(glob(os.path.join(train_path, 'data', '*.npy')))
            target_path = sorted(glob(os.path.join(train_path, 'label', '*.npy')))
            input_ = [f for f in input_path]
            target_ = [f for f in target_path]
            if load_mode == 0:  # batch data load
                self.input_ = input_
                self.target_ = target_
            else:  # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
        else:  # mode =='test'
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            if load_mode == 0:  # batch data load
                self.input_ = input_
                self.target_ = target_
            else:    # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
        

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx):
        input_img, target_img = self.input_[idx], self.target_[idx]
        if self.load_mode == 0:
            input_img, target_img = np.load(input_img), np.load(target_img)
        
        # do normalization
        if self.norm:
            input_mean, input_std = np.mean(input_img), np.std(input_img)
            target_mean, target_std = np.mean(target_img), np.std(target_img)
            input_img = (input_img - input_mean) / input_std
            target_img = (target_img - target_mean) / target_std

        # if self.mode == 'train' and self.transform:
        #     input_img = self.preprocess(input_img)
        #     target_img = self.preprocess(target_img)


        if self.mode == 'train' and self.patch_training:
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            return (input_patches, target_patches)
        else:
            return (input_img, target_img)


def get_patch(full_input_img, full_target_img, patch_n, patch_size): # 定义patch
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        top = np.random.randint(0, h-new_h)
        left = np.random.randint(0, w-new_w)
        patch_input_img = full_input_img[top:top+new_h, left:left+new_w]
        patch_target_img = full_target_img[top:top+new_h, left:left+new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)


def get_loader(mode='train', load_mode=0, train_path = None,
               saved_path=None, test_patient='LDCT',
               patch_n=None, patch_size=None,
               transform=None, batch_size=32, 
               num_workers=6, norm = False, patch_training = False):
    train_dataset= ct_dataset('train', load_mode, train_path, saved_path, test_patient, patch_n, patch_size, transform, norm, patch_training)
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  

    test_dataset = ct_dataset('test', load_mode, train_path, saved_path, test_patient, patch_n, patch_size, transform, norm, False)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)  
    return train_data_loader, test_data_loader
