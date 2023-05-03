import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from configs.road_segnet_more_data_hsv_20230420 import model

from datasets.base_dataset import BaseDataset
import albumentations as A
from losses.dice_loss import CE_DiceLoss

def whole_window_padding(image, window_size):
    w, h = image.shape[0], image.shape[1]
    ww, wh = window_size[0], window_size[1]
    if ww >= w or wh >= h:
        w_diff = ww - w
        h_diff = wh - h
        
    else:
        w_diff = ww - w % ww
        h_diff = wh - h % wh
    _top =  w_diff // 2
    _bottom = w_diff - w_diff // 2
    _left = h_diff // 2
    _right = h_diff - h_diff // 2
    img = cv2.copyMakeBorder(image, top = _top, bottom = _bottom,
                                 left = _left, right = _right,
                                 borderType = cv2.BORDER_CONSTANT, value = 0)
    print(img.shape)
    return img, _left, _left + image.shape[1], _top, _top + image.shape[0]

def infer(model, image, resolution, device):
    image_padded, ratiox, ratioy, pad_left, pad_top = BaseDataset.pad_image_to_input(image, resolution=resolution, ret_ratio_offset = True, random_pad=False)
    image_in = BaseDataset.convert_to_image_input(image_padded)
    data_in = np.array([image_in])
    data_in = torch.from_numpy(data_in)
    data_in.to(device)
    data_out = torch.sigmoid(model.forward(data_in))
    masks = data_out.detach().cpu().numpy()
    mask = masks[0][0][pad_top: pad_top + image.shape[0], pad_left: pad_left + image.shape[1]]
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation = 1)
    return mask

def SlideWindowInference(model, raw, size_model_need, window_size, window_stride, raw_padding = False, mode = 'avg', device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # window_size could be not equal to size_model_need
    if raw_padding:
        raw, rtop, rbottom, rleft, rright = whole_window_padding(raw, window_size)
        print(rleft, rright, rtop, rbottom)
    result = np.zeros((raw.shape[0], raw.shape[1]))
    compute_times = np.zeros((raw.shape[0], raw.shape[1]))
    plt.figure('raw_padding')
    plt.imshow(raw)
    for x in np.arange(0, raw.shape[0], window_stride[0]):
        for y in np.arange(0, raw.shape[1], window_stride[1]):
            right = min(raw.shape[0], x + window_size[0])
            bottom = min(raw.shape[1], y + window_size[1])
            window = raw[x: right, y: bottom]
            # print(x, right, y, bottom)
            # plt.figure('raw')
            # plt.imshow(window)
            tmp = infer(model, window, size_model_need, device)

            if mode == 'avg':
                result[x: right, y: bottom] += tmp
                compute_times[x: right, y: bottom] += 1

            elif mode == 'max':
                result[x: right, y: bottom] = np.maximum(result[x: right, y: bottom], tmp)
                
            # if bottom == raw.shape[1]:
            #     plt.figure('raw')
            #     plt.imshow(window)
            #     plt.figure('tmp')
            #     plt.imshow(tmp)
            #     plt.colorbar()
            #     plt.show()
    if mode == 'avg':
        compute_times[compute_times == 0] = 1
        result = result / compute_times
    if raw_padding:
        return result[rleft: rright, rtop: rbottom]
    else:
        return result

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load('test.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model.to(device)
    image = cv2.imread('test.png')
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    



    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.copyMakeBorder(image, top=500, bottom=500,left=500,right=500,borderType=cv2.BORDER_CONSTANT,value=0)
    # image = cv2.convertScaleAbs(image, alpha=0.5, beta=0)
    image_padded = BaseDataset.pad_image_to_input(image, resolution=(1024, 1024), random_pad=False)
    image_in = BaseDataset.convert_to_image_input(image_padded)
    data_in = torch.from_numpy(np.array([image_in])).to(device)
    data_out = torch.sigmoid(model.forward(data_in))
    masks = data_out.detach().cpu().numpy()
    mask = masks[0]


    plt.figure('raw')
    plt.imshow(image_padded)
    plt.figure("normal_Mask")
    for mask_for_cls in mask:
        plt.imshow(mask_for_cls)

    plt.show()


    thres = 0.7
    mask = cv2.resize(mask[0], (image.shape[1], image.shape[0]), interpolation=1)
    mask = mask >= thres
    image_t = image.copy()
    image_t[~mask] = 0
    plt.figure(f"Normal Method Threshold {thres}")
    plt.imshow(image_t)

    slide_window_mask = SlideWindowInference(model, image, (1024, 1024), (1024, 1024), (1024, 1024), raw_padding = True, mode = 'max')
    print(slide_window_mask.shape)
    plt.figure('raw')
    plt.imshow(image)


    plt.figure('slide_window_mask')
    plt.imshow(slide_window_mask)
    plt.colorbar()

    plt.figure('slide_window_result')
    plt.imshow(image, alpha=1)
    plt.imshow(slide_window_mask, alpha=0.5)
    plt.colorbar()
    


    mask = slide_window_mask >= thres
    image_t = image.copy()
    image_t[~mask] = 0

    plt.figure(f"Slide Window Threshold {thres}")
    plt.imshow(image_t)

    plt.show()