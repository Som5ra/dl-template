import logging
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import albumentations as A
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class BaseDataset(Dataset):
    IMG_NORM_MEAN = np.array([0.40789654, 0.44719302, 0.47026115]).reshape((3, 1, 1))
    IMG_NORM_STD = np.array([0.28863828, 0.27408164, 0.27809835]).reshape((3, 1, 1))

    def __init__(self, network_resolution, augmentation_pipeline = None) -> None:
        super().__init__()
        self.network_resolution = network_resolution
        self.augmentation_pipeline = augmentation_pipeline
        if isinstance(self.augmentation_pipeline, str):
            self.augmentation_pipeline = A.load(self.augmentation_pipeline)
        
        self.debug = os.environ.get('DEBUG_DATASET_SHOW_IMG', '').lower() == 'true'

    
    def _get_image_mask_from_index(self, index):
        '''
        Return (image_rgb, mask)

        image_rgb: (W, H, 3)
        mask: (W, H, C)
        '''
        raise NotImplementedError()
    

    def __getitem__(self, index):
        image_rgb, mask = self._get_image_mask_from_index(index) 
        if self.augmentation_pipeline is not None:
            transformed = self.augmentation_pipeline(image=image_rgb, mask=mask)
            image_rgb = transformed['image']
            mask = transformed['mask']
        # plt.figure('image')
        # plt.imshow(image_rgb)

        # plt.figure('mask')
        # plt.imshow(mask)

        image_rgb = BaseDataset.pad_image_to_input(image_rgb, resolution=self.network_resolution, random_pad=False)

        # plt.figure('image_padded')
        # plt.imshow(image_rgb)

        mask = mask.astype(np.float32)
        mask = BaseDataset.pad_image_to_input(mask, resolution=self.network_resolution, random_pad=False)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1

        if self.debug:
            plt.figure('image_padded')
            plt.imshow(image_rgb)

            plt.figure('mask_padded')
            plt.imshow(mask)
            plt.show()

        image_rgb = BaseDataset.convert_to_image_input(image_rgb)
        mask = BaseDataset.convert_to_mask_input(mask)

        return torch.from_numpy(image_rgb), torch.from_numpy(mask)


    @staticmethod
    def convert_to_image_input(image_rgb):
        image_norm = np.array(image_rgb, dtype=np.float32).transpose(2, 0, 1)
        image_norm /= 255.0
        image_norm -= BaseDataset.IMG_NORM_MEAN
        image_norm /= BaseDataset.IMG_NORM_STD
        return image_norm
    
    @staticmethod
    def convert_to_mask_input(mask):
        mask = np.array(mask, dtype=np.float32).transpose(2, 0, 1)
        return mask
    
    @staticmethod
    def pad_image_to_input(image_in, resolution, ret_ratio_offset = False, random_pad = False, scale = True, interpolation = 3):
        '''
        Pad image to targeted size (preserve ratio)
        '''
        net_w, net_h = resolution
        h, w, c = image_in.shape

        wh_ratio = w/h
        ratiox = 1.0
        ratioy = 1.0

        if scale:
            if net_h * wh_ratio > net_w:
                ratiox = net_w / w
                ratioy = net_w / wh_ratio / h
            else:
                ratioy = net_h / h
                ratiox = net_h * wh_ratio / w

        new_h = int(h * ratioy)
        new_w = int(w * ratiox)


        pad_top = 0 if not random_pad else random.randint(0, net_h - new_h)
        pad_bottom = net_h - new_h - pad_top
        pad_left = 0 if not random_pad else random.randint(0, net_w - new_w)
        pad_right = net_w - new_w - pad_left

        image_paded = cv2.resize(image_in, (new_w, new_h), interpolation=interpolation)
        image_paded = cv2.copyMakeBorder(image_paded, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value = 0)

        if len(image_paded.shape) == 2:
            image_paded = np.expand_dims(image_paded, -1)
        if ret_ratio_offset:
            return image_paded, ratiox, ratioy, pad_left, pad_top
        return image_paded


if __name__ == "__main__":
    mask = np.random.rand(512, 512, 2) > 0.5
    mask = mask.astype(np.float32)
    mask_padded = BaseDataset.pad_image_to_input(mask, resolution=(1024, 1024))
    print(mask, mask.shape)
    print(mask_padded, mask_padded.shape)