import torch
from utils.Dataset import Dataset


if __name__ == '__main__':
    dataset = Dataset(mode='val', dataset_dir_paths=['./DamDataset'], patch_size=64)
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0)
    for a, b, label in loader:
        print(f"==>> label: {label}")