from itertools import accumulate
from torch.utils.data import Dataset

class ConcatDataset(Dataset):

    def __init__(self, datasets, ) -> None:
        super().__init__()
        self.datasets = datasets
        self.datasets_len = list(map(lambda x: len(x), self.datasets))
        self.datasets_cum_len = list(accumulate(self.datasets_len))


    def __len__(self,):
        return sum(map(lambda x: len(x), self.datasets))
    

    def __getitem__(self, index):
        dataset_idx = None
        index_offset = None
        for i, len in enumerate(self.datasets_cum_len):
            if index < len:
                dataset_idx = i
                index_offset = 0 if i == 0 else self.datasets_cum_len[i - 1]
                break
        
        print(self.datasets_cum_len)
        print("dataset_idx", dataset_idx)
        print("index", index)
        print("index_offset", index_offset)
        return self.datasets[dataset_idx].__getitem__(index - index_offset)

if __name__ == "__main__":
    from datasets.labelme_dataset import LabelmeDataset

    dataset = ConcatDataset(
        [
            LabelmeDataset(
                network_resolution=(512, 512),
                labelme_folders = [
                    "data/coco2014_midrange_maskout"
                ],
                classes=['human'],
            ),
            LabelmeDataset(
                network_resolution=(512, 512),
                labelme_folders = [
                    "data/coco2014_midrange_maskout"
                ],
                classes=['human'],
            )
        ]
    )