import os
import json
import cv2
import random
import numpy as np
from pathlib import Path
from datasets.base_dataset import BaseDataset
# from base_dataset import BaseDataset
import albumentations as A

class MaskImageDataset(BaseDataset):
    '''
    image_masks_folders = [
        [
            "PATH TO IMAGE",
            [
                "PATH TO CLASS 1 MASK",
                "PATH TO CLASS 2 MASK",
                ...
            ]
        ]
    ]
    '''
    def __init__(self, network_resolution, image_masks_folders, augmentation_pipeline=None, exts = ['png', 'jpeg', 'jpg', 'bmp']) -> None:
        super().__init__(network_resolution, augmentation_pipeline)
        self.image_masks_folders = image_masks_folders
        self.classes = len(image_masks_folders[0][1])
        self.exts = exts

        # get images
        self.image_masks_pairs = []
        for image_folder, mask_folders in image_masks_folders:
            files = list(map(lambda x: str(x), Path(image_folder).glob('*.*')))
            files = list(filter(lambda x: x.rsplit('.', 1)[-1] in exts, files ))
            for image_file in files:
                mask_files = []
                for mask_folder in mask_folders:
                    mask_file = os.path.join(mask_folder, os.path.basename(image_file))
                    possible_mask_files = [ os.path.join(mask_folder, os.path.basename(image_file).rsplit('.', 1)[0] + f".{ext}") for ext in exts ]

                    found = False
                    for mask_file in possible_mask_files:
                        if os.path.exists(mask_file):
                            mask_files.append(mask_file)
                            found = True
                            break


                    if not found:
                        raise Exception(f"Expected mask file not found: {mask_file}")
            
                self.image_masks_pairs.append((image_file, mask_files))

    def __len__(self,):
        return len(self.image_masks_pairs)
    
    def _get_image_mask_from_index(self, index):
        image_file, mask_files = self.image_masks_pairs[index]

        image_rgb = cv2.imread(image_file)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        maskArr = np.zeros((image_rgb.shape[0], image_rgb.shape[1], self.classes), np.uint8)
        
        for cls_id, mask_file in enumerate(mask_files):
            image_mask = cv2.imread(mask_file)
            maskArr[:, :, cls_id] = image_mask[:, :, 1] > 0

        return image_rgb, maskArr

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    training_set = MaskImageDataset(
        network_resolution=(1024, 1024),
            image_masks_folders = [
                [
                    "/home/risksis/highway_pcd_image_test/road_detection/data/dataset/images",
                    [
                        "/home/risksis/highway_pcd_image_test/road_detection/data/dataset/mask"
                    ],
                ],
                [
                    "/home/risksis/highway_pcd_image_test/road_detection/data/dataset/images",
                    [
                        "/home/risksis/highway_pcd_image_test/road_detection/data/dataset/mask"
                    ],
                ]
            ],
            augmentation_pipeline=A.Compose([
                A.HueSaturationValue(always_apply=False, p=1.0, hue_shift_limit=(-40, 40), sat_shift_limit=(-40, 40), val_shift_limit=(0, 50)),
        
                # A.OneOf([
                #     A.RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(-0.1, 0.1), contrast_limit=(0, 0), brightness_by_max=True),
                # ], p=1),
                A.RandomRotate90(always_apply=False, p=1.0),
                A.Flip(always_apply=False, p=1.0),
                # A.GaussNoise(always_apply=False, p=0.5, var_limit=(0.0, 100.0), per_channel=True, mean=0.0),
                
                # A.CoarseDropout(always_apply=False, p=0.5, max_holes=8, max_height=8, max_width=8, min_holes=8, min_height=8, min_width=8, fill_value=(0, 0, 0), mask_fill_value=None),
                # A.PixelDropout(always_apply=False, p=0.25, dropout_prob=0.1, per_channel=0, drop_value=(0, 0, 0), mask_drop_value=None),
                A.PixelDropout(always_apply=False, p=0.25, dropout_prob=0.2, per_channel=0, drop_value=(0, 0, 0), mask_drop_value=None),
                A.ElasticTransform(always_apply=False, p=0.5, alpha=0.0, sigma=0.0, alpha_affine=100.0, interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, approximate=False, same_dxdy=False),
                A.ShiftScaleRotate(always_apply=False, p=1.0, shift_limit_x=(-0.1, 0.1), shift_limit_y=(-0.1, 0.1), scale_limit=(-0.3, 0.3), rotate_limit=(-10, 10), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None, rotate_method='largest_box'),
                A.OneOf([
                    A.RandomCrop(512, 512, always_apply = False, p = 1),
                    A.RandomCrop(768, 768, always_apply = False, p = 1),
                    A.RandomCrop(1024, 1024, always_apply = False, p = 1),
                    A.RandomCrop(2048, 2048, always_apply = False, p = 1),
                ], p=1),
                # A.RandomCrop(512, 832, always_apply = False, p = 1),
        ])
    )

    for i in range(training_set.__len__()):
        image, mask = training_set.__getitem__(i)
