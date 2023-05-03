from configs.base_config import *

import albumentations as A
from datasets.labelme_dataset import LabelmeDataset
from datasets.mask_image_dataset import MaskImageDataset
from losses.focal_loss import focal_loss
from losses.dice_loss import CE_DiceLoss
from models.backbones.resnet101 import ResNet101
from models.backbones.resnet_fpn import ResNetFPN
from models.heads.deconv_v1 import DeconvV1
from models.unet import UNet
from models.segnet import UNetResNet
import torch.optim as optim
import torch.nn as nn
import random

# data
training_set = MaskImageDataset(
    network_resolution=(1024, 1024),
        image_masks_folders = [
            [
                './test_images'
                [
                    './test_masks'
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

# model
# model = DeconvV1(
#     backbone=ResNetFPN(ResNet101(), outplanes=1024)
# )
# model = UNet(in_channels = 3, out_channels = 1, init_features = 32)
model = UNetResNet(encoder_depth=101, num_classes=1)
pretrained_model_path = None
# loss = focal_loss
# loss = CE_DiceLoss
loss = nn.BCEWithLogitsLoss()

# training
model.train()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[140, 170], gamma=0.1)



fp16 = True