import os
import json
import cv2
import numpy as np
from pathlib import Path
from datasets.base_dataset import BaseDataset


class LabelmeDataset(BaseDataset):
    def __init__(self, network_resolution, labelme_folders, classes, augmentation_pipeline=None) -> None:
        super().__init__(network_resolution, augmentation_pipeline)
        self.labelme_folders = labelme_folders
        self.classes = classes

        # get images
        self.image_json_pairs = []
        for labelme_folder in labelme_folders:
            files = list(map(lambda x: str(x), Path(labelme_folder).glob('*.*')))
            files = list(filter(lambda x: x.rsplit('.', 1)[-1] in ['png', 'jpeg', 'jpg', 'bmp'], files ))

            for image_file in files:
                json_file = image_file.rsplit('.', 1)[0] + '.json'
                if os.path.exists(json_file):
                    self.image_json_pairs.append([ image_file, json_file ])

    
    def __len__(self,):
        return len(self.image_json_pairs)
    
    def _get_image_mask_from_index(self, index):
        image_file, json_file = self.image_json_pairs[index]

        image_rgb = cv2.imread(image_file)
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)

        with open(json_file, 'r') as fp:
            labelme_data = json.load(fp)

        maskArr = np.zeros((image_rgb.shape[0], image_rgb.shape[1], len(self.classes)), np.uint8)

        for cls_id, cls in enumerate(self.classes):
            for shape in filter(lambda x: x['label'] in ( cls if isinstance(cls, list) else [cls] ), labelme_data.get('shapes', [])):
                # cv2Pts = []
                # for point in shape['points']:
                #     x, y = point
                #     cv2Pts.append((int(x), int(y)))
                cv2Pts = np.array(shape['points'], dtype=np.int32)
                cv2.fillPoly(maskArr[:, :, cls_id:cls_id+1], [cv2Pts], 1)


        return image_rgb, maskArr

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = LabelmeDataset(
        network_resolution=(512, 512),
        labelme_folders=[
            '/home/risksis/highway_pcd_image_test/tactile_segmentation/data/google_map_photos_20230309'
        ],
        classes=['tactile']
    )

    for i in range(dataset.__len__()):
        image, mask = dataset._get_image_mask_from_index(i)
        plt.figure()
        plt.imshow(image)
        plt.figure()
        plt.imshow(mask)

        plt.show()
        