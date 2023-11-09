import torch
from utils.Dataset import Dataset
from tqdm import tqdm

if __name__ == '__main__':
    dataset = Dataset(mode='train', dataset_dir_paths=['./DamDataset/dataV1'], patch_size=64)
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, num_workers=0)
    
    nb_samples = 0.
    channel_mean = torch.zeros(3)
    channel_std = torch.zeros(3)
    
    for a, b, label in tqdm(loader, ncols=100):
        N, C, H, W = a.shape[:4]
        data = a.view(N, C, -1)

        channel_mean += data.mean(2).sum(0)
        channel_std += data.std(2).sum(0)
        nb_samples += N
        
        N, C, H, W = b.shape[:4]
        data = b.view(N, C, -1)

        channel_mean += data.mean(2).sum(0)
        channel_std += data.std(2).sum(0)
        nb_samples += N

    channel_mean /= nb_samples
    channel_std /= nb_samples
    print(channel_mean, channel_std)
        