import math
import os
from typing import Tuple
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from PIL import Image, ImageOps

def squash_packed(x, fn):
    return torch.nn.utils.rnn.PackedSequence(fn(x.data), x.batch_sizes, x.sorted_indices, x.unsorted_indices)

def pack_sequence_with_batch_first(sequence):
    """将sequence列表转化为packed sequence

    Args:
        sequence (list): 由sequence组成的列表

    Returns:
        PackedSequence: 包装好的数据
    """    
    
    seqs_lens = [s.size(0) for s in sequence]
    sequence = pad_sequence(sequence, batch_first=True)    
    sequence = pack_padded_sequence(sequence, seqs_lens, batch_first=True, enforce_sorted=False)
    
    return sequence

def unpack_sequence_with_batch_first(packed_sequence):
    """将PackedSequence进行解包

    Args:
        packed_sequence (PackedSequence): 打包好的序列数据

    Returns:
        list[Tensor]: 解包还原成原样的list序列数据
    """    
    
    sequences, sequence_lens = pad_packed_sequence(packed_sequence, batch_first=True)
    output = [sequences[i][:sequence_lens[i]] for i in range(len(sequence_lens))]
    
    return output

def imread(filepath: str, flags: int, patch_size: int, expand: bool=True) -> np.ndarray: 
    """读取图像, 三通道的通道顺序为RGB

    Args:
        filepath (str): 文件路径
        flags (int): 读取模式, 0为灰度模型，1为RGB模式

    Returns:
        np.ndarray: 返回读取并处理后的图像
    """
    # 针对大坝数据集新标注的特判
    if not os.path.exists(filepath):
        filepath = filepath[:-3] + 'JPG'
    if not os.path.exists(filepath):
        filepath = filepath[:-3] + 'png'
    if flags == 0:
        image = Image.open(filepath).convert("L")
    elif flags == -1:
        image = Image.open(filepath).convert("RGB")
    assert image is not None, f'PathError: {filepath}'
    image = ImageOps.exif_transpose(image)
    image = np.array(image)
    # print(f"==>> image.shape: {image.shape}")

    if expand:
        image, (top, left, h, w) = expand_image(image, patch_size)
        return image, (top, left, h, w)

    return image, None

def expand_image(img: np.ndarray, patch_size: int) -> np.ndarray:
    """用padding扩充图像

    Args:
        img (np.ndarray): 原图像
        new_size (tuple): 新的尺寸

    Returns:
        np.ndarray: 扩充之后的图像
    """
    # 图像尺寸不为滑窗尺寸(经过权衡之后选择滑窗尺寸, 而不是切片尺寸)的倍数, 则用黑边补全
    def cal_new_lengh(x, p, i): return math.ceil((x-p)/i)*i+p
    h, w, *_ = img.shape
    new_size = (cal_new_lengh(h, patch_size, patch_size),
                cal_new_lengh(w, patch_size, patch_size))
    
    new_h, new_w, *_ = new_size
    img_h, img_w, *_ = img.shape

    top = (new_h - img_h) // 2
    bottom = new_h - img_h - top
    left = (new_w - img_w) // 2
    right = new_w - img_w - left

    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)), (top, left, h, w)

def get_patch_num(image: np.ndarray, patch_size) -> Tuple[int, int]:
    """返回image在高和宽上分别能够切的滑窗个数

    Args:
        image (np.ndarray): 输入图像

    Returns:
        tuple[int, int]: 在高和宽上分别能够切的滑窗个数
    """
    h, w, *_ = image.shape

    patch_num_w = math.ceil((w - patch_size) / patch_size) + 1
    patch_num_h = math.ceil((h - patch_size) / patch_size) + 1

    if w - patch_size < 0:
        patch_num_w = 0
    if h - patch_size < 0:
        patch_num_h = 0

    return patch_num_h, patch_num_w        

def get_box(patch_size: int, x: int, y: int) -> Tuple[int, int, int, int]:
    
    xmin = x * patch_size
    ymin = y * patch_size
    xmax = (x + 1) * patch_size
    ymax = (y + 1) * patch_size
    
    return xmin, ymin, xmax, ymax

def get_patch(image: np.ndarray, x: int, y: int, patch_size:int):
    
    xmin = x * patch_size
    ymin = y * patch_size
    xmax = xmin + patch_size
    ymax = ymin + patch_size
    
    return image[ymin:ymax, xmin:xmax]

class WarmupCosineSchedule(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(WarmupCosineSchedule, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            lr_mult = float(epoch) / float(max(1, self.warmup_epochs))
        else:
            progress = float(epoch - self.warmup_epochs) / float(max(1, self.total_epochs - self.warmup_epochs))
            lr_mult = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return [max(self.min_lr, base_lr * lr_mult) for base_lr in self.base_lrs]

def get_prompt(filename, root):
    try:
        box_path = os.path.join(root, f"{filename.split('.')[0]}.txt")
        box = list(map(int, open(box_path).readlines()[0].split()[:4]))
        
        return box
    except:
        return [0, 0, 64, 64]

def get_dataset_pathes(dir_path:str, mode:str):
    """
    获取数据集的文件路径

    Parameters
    ----------
    dir_path : str
        数据集根目录
    mode : str
        数据集子集，train or val

    Returns
    -------
    tuple[list, list]
        图像文件路径列表，标注文件路径列表
    """
    if os.path.exists(os.path.join(dir_path, 'train.txt')):
        train_filenames = [i.replace('\n', '') for i in open(os.path.join(dir_path, 'train.txt')).readlines()]
        val_filenames = [i.replace('\n', '') for i in open(os.path.join(dir_path, 'val.txt')).readlines()]
    else:
        filenames = sorted(os.listdir(os.path.join(dir_path, 'image')))
        train_filenames, val_filenames = train_test_split(filenames,
                                                        test_size=0.2,
                                                        random_state=42)
        for mode in ['train', 'val']:
            with open(os.path.join(dir_path, f'{mode}.txt'), "w+") as file:
                for i in eval(f'{mode}_filenames'):
                    file.write(i+'\n')
    
    dataset_filenames = {
        'train': train_filenames,
        'val': val_filenames
    }
    
    image_paths = []
    label_paths = []
    
    image_paths += [
        os.path.join(dir_path, 'image', i)
        for i in dataset_filenames[mode]
    ]
    label_paths += [
        os.path.join(dir_path, 'label', i)
        for i in dataset_filenames[mode]
    ]
    
    return image_paths, label_paths

def cal_crack_size(label_image: np.ndarray, xmin: int, ymin: int,
                    xmax: int, ymax: int) -> float:
    """
    计算裂缝切片像素个数

    Parameters
    ----------
    label_image : np.ndarray
        原标注图像
    xmin : int
        左上角x坐标
    ymin : int
        左上角y坐标
    xmax : int
        右下角x坐标
    ymax : int
        右下角y坐标

    Returns
    -------
    float
        裂缝像素个数
    """
    h, w = label_image.shape

    if xmin < 0 or ymin < 0 or xmax > w or ymax > h:
        return 0

    box = label_image[ymin:ymax, xmin:xmax]
    
    return box.sum() // 255

def judge_crack(label_image:np.ndarray, x:int, y:int, patch_size:int):
    
    xmin = x * patch_size
    ymin = y * patch_size
    xmax = xmin + patch_size
    ymax = ymin + patch_size
    
    return cal_crack_size(label_image, xmin, ymin, xmax, ymax) >= 40