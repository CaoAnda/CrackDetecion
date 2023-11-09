from multiprocessing import Manager, Process
from PIL import Image
from colorama import Fore
import albumentations as A
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import cv2
from torch.utils import data
import torch
import numpy as np
import os
from functools import partial
from tqdm import tqdm
import sys
from queue import Queue
from collections import deque
from .tools import get_box, get_patch_num, imread, expand_image, get_patch
import random

tqdm = partial(tqdm, ncols=100, file=sys.stdout)

Image.MAX_IMAGE_PIXELS = 150000000

class Dataset(data.Dataset):

    def __init__(self,
                 mode: str,
                 dataset_dir_paths: list[str],
                 patch_size: int = 64,
                 enhanced=0,
                 **args) -> None:
        """
        数据集

        Parameters
        ----------
        mode : str
            数据集的类型，'train' or 'val'
        dataset_dir_paths : List[str]
            数据集的根路径
        patch_size : int, optional
            切片大小, by default 64
        seq_len : int, optional
            生成的序列长度, by default 8
        enhanced : bool, optional
            是否进行数据增强, by default False
        """
        super().__init__()
        self.image_paths = []
        self.label_paths = []
        self.mode = mode
        self.patch_size = patch_size
        self.dataset_dir_paths = dataset_dir_paths
        self.images = []
        self.labels = []
        self.enhanced_images = []
        self.enhanced_labels = []
        self.pair_list = [] # obejct: pic-id, [x, y], [x, y], sim
        self.enhanced = False if mode == 'val' else enhanced
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.6121, 0.6286, 0.6258),
                                 std=(0.0358, 0.0367, 0.0370)),
        ])
        self.enhanced_thread = None
        # 使用不同的变量，尝试解决死锁问题
        self._enhanced_images = []
        self._enhanced_labels = []
        self._pair_list = []
        self.spatial_transform = A.Compose([
            # A.RandomResizedCrop(patch_size * 20, patch_size * 20, scale=(0.5, 1)),
            A.HorizontalFlip(),
            A.ShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=.1),
                    # A.PiecewiseAffine(p=0.3),
                    ], p=0.2)
            ],
                additional_targets={
                'label': 'image'
            }
        )
        self.pixel_transform = A.Compose([
            A.GaussNoise(p=0.2),
            A.OneOf([
                    A.MotionBlur(p=.2),
                    A.MedianBlur(blur_limit=3, p=0.1),
                    A.Blur(blur_limit=3, p=0.1),
                    ], p=0.2),
            A.OneOf([
                    A.CLAHE(clip_limit=2),
                    A.Sharpen(),
                    A.Emboss(),
                    A.RandomBrightnessContrast(),
                    ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
        self.init_path()
        self.image_init()

    def init_path(self):
        """初始化图片路径
        """
        # 验证集只使用大坝的验证集，不额外添加
        # if self.mode == 'val':
        #     self.dataset_dir_paths = ['./DamDataset/dataV1']

        for dir_path in self.dataset_dir_paths:
            if os.path.exists(os.path.join(dir_path, 'train.txt')):
                train_filenames = [i.replace('\n', '') for i in open(os.path.join(dir_path, 'train.txt')).readlines()]
                val_filenames = [i.replace('\n', '') for i in open(os.path.join(dir_path, 'val.txt')).readlines()]
            else:
                filenames = sorted(os.listdir(os.path.join(dir_path, 'image')))
                train_filenames, val_filenames = train_test_split(filenames,
                                                              test_size=0.2,
                                                              random_state=42)
            
            dataset_filenames = {
                'train': train_filenames,
                'val': val_filenames
            }
            self.image_paths += [
                os.path.join(dir_path, 'image', i)
                for i in dataset_filenames[self.mode]
            ]
            self.label_paths += [
                os.path.join(dir_path, 'label', i)
                for i in dataset_filenames[self.mode]
            ]
            
        print(f'mode: {self.mode}, dataset length: {len(self.image_paths)}')

    def image_init(self):
        """初始化图像
        """
        # self.image_paths = self.image_paths[:30]
        # self.label_paths = self.label_paths[:30]
        for index in tqdm(range(len(self.image_paths)),
                          desc=f'{self.mode} Loading Index List'):
            image, *_ = imread(self.image_paths[index], -1, self.patch_size, expand=not self.enhanced)
            label, *_ = imread(self.label_paths[index], 0, self.patch_size, expand=not self.enhanced)
            _, label = cv2.threshold(label, 10, 255, cv2.THRESH_BINARY)
            self.images.append(image)
            self.labels.append(label)
            assert image.shape[:2] == label.shape, f'image.shape is {image.shape}, label.shape is {label.shape}'
            if not self.enhanced:
                self.update_patch_pair_from_label(index, label)
        if not self.enhanced:
            random.shuffle(self._pair_list)
            self.pair_list = self._pair_list
        if self.enhanced:
            self.start_enhanced_thread(0)
    
    def enhance_image(self, image: np.ndarray, label: np.ndarray, enhanced: int) -> tuple[np.ndarray, np.ndarray]:
        """图像增强

        Args:
            image (np.ndarray): 原图像
            label (np.ndarray): 标注图像

        Returns:
            tuple[np.ndarray, np.ndarray]: 增强后的图像
        """
        # 由PIL读入图像默认为RGB
        if enhanced == 1 or enhanced == 3:
            augmented_result = self.spatial_transform(image=image, label=label)
            image = augmented_result['image']
            label = augmented_result['label']
        if enhanced == 2 or enhanced == 3:
            image = self.pixel_transform(image=image)['image']
        return image, label
    
    def access_pre_setting(self, epoch: int):
        """
        开始当前epoch的预设置，主要与数据增强相关变量有关

        Parameters
        ----------
        epoch : int
            当前epoch编号
        """ 
        self.enhanced_images = list(self._enhanced_images).copy()
        self.enhanced_labels = list(self._enhanced_labels).copy()
        self.pair_list = list(self._pair_list).copy()
    
    def update_patch_pair_from_label(self, index: int, label_image: np.ndarray):
        """
        根据label生成对应的图像pair对

        Parameters
        ----------
        index : int
            当前label图像对应的index
        label_image : np.ndarray
            label图像
        """
        patch_size = self.patch_size
        h, w = get_patch_num(label_image, patch_size)
        crack_patch_list = []
        background_patch_list = []
        
        for y in range(h):
            for x in range(w):
                xmin = x * patch_size
                ymin = y * patch_size
                xmax = xmin + patch_size
                ymax = ymin + patch_size
                if self.cal_crack_size(label_image, xmin, ymin, xmax, ymax) >= 40:
                    crack_patch_list.append([x, y])
                else:
                    background_patch_list.append([x, y])
        
        len_crack_patch_list = len(crack_patch_list)
        len_background_patch_list = len(background_patch_list)
        length = min(len_crack_patch_list, len_background_patch_list)
        sample_times = 5
        for _ in range(length*sample_times):
            
            get_pair = lambda index, lista, listb, sim: {
                'pic_id': index,
                'a': lista[random.randint(0, len(lista)-1)],
                'b': listb[random.randint(0, len(listb)-1)],
                'sim': sim
            } 
            
            self._pair_list.append(get_pair(index, crack_patch_list, crack_patch_list, 1))
            self._pair_list.append(get_pair(index, crack_patch_list, background_patch_list, 0))
            self._pair_list.append(get_pair(index, background_patch_list, crack_patch_list, 0))
            self._pair_list.append(get_pair(index, background_patch_list, background_patch_list, 1))
        
    def update_enhanced_images(self, epoch: int, enhanced: int):
        """更新数据增强的图像

        Args:
            epoch (int): 数据增广对应的epoch id
        """
        import random
        random.seed(epoch)
        for index in range(len(self.image_paths)):
            enhanced_image, enhanced_label = self.enhance_image(
                self.images[index], self.labels[index], enhanced)
            # cv2.imwrite('test.png', enhanced_image[:, :, ::-1])
            enhanced_image, *_= expand_image(enhanced_image, self.patch_size)
            enhanced_label, *_ = expand_image(enhanced_label, self.patch_size)
            self._enhanced_images.append(enhanced_image)
            self._enhanced_labels.append(enhanced_label)
            self.update_patch_pair_from_label(index, enhanced_label)
        random.shuffle(self._pair_list)
    
    def start_enhanced_thread(self, epoch: int):
        """
        启动epoch对应的数据增强进程

        Parameters
        ----------
        epoch : int
            epoch的编号
        """        
        print(Fore.LIGHTBLACK_EX + f'Start Loading Enhanced Images of Epoch-{epoch}...')
        self._enhanced_images = Manager().list()
        self._enhanced_labels = Manager().list()
        self._pair_list = Manager().list()
        self.enhanced_thread = Process(
            target=self.update_enhanced_images, args=(epoch, self.enhanced))
        # 设置为守护进程，当父进程结束时，该进程也会自动结束
        self.enhanced_thread.daemon = True
        self.enhanced_thread.start()
    
    def get_enhanced_process(self, epoch: int):
        """获取数据增广当前进度

        Args:
            epoch (int): 数据增广对应epoch id

        Returns:
            str: 表示数据增广进度的字符串.
        """
        return f"{len(self._enhanced_images)} / {len(self.image_paths)}"

    def cal_crack_size(self, label_image: np.ndarray, xmin: int, ymin: int,
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

    def draw_rectangles(self, image: np.ndarray,
                        rectangles: list) -> np.ndarray:
        """
        在原图上标注出对应的切片list

        Parameters
        ----------
        image : np.ndarray
            原图
        rectangles : list
            切片list的box坐标

        Returns
        -------
        np.ndarray
            结果图像
        """
        # 复制图像，以免修改原始图像
        output_image = image.copy()

        # 遍历矩形列表
        for node in rectangles:
            xmin, ymin, xmax, ymax = get_box(self.patch_size, node['x'],
                                             node['y'])

            # 在图像上绘制矩形
            cv2.rectangle(output_image, (xmin, ymin), (xmax, ymax),
                          (0, 255, 0), 2)  # 可根据需要修改颜色和线宽

        # 返回绘制了矩形的图像
        return output_image
    
    def __getitem__(self, index: int):
        images = self.enhanced_images if self.enhanced else self.images 
        # labels = self.enhanced_labels if self.enhanced else self.labels 
        pair = self.pair_list[index]
        image = images[pair['pic_id']]
        # label = labels[pair['pic_id']]
        patch_A = get_patch(image, *pair['a'], self.patch_size)
        patch_B = get_patch(image, *pair['b'], self.patch_size)
        # label_A = get_patch(label, *pair['a'])
        # label_B = get_patch(label, *pair['b'])
        
        # sizeA = label_A.sum() / 255
        # print(f"==>> sizeA: {sizeA}")
        # sizeB = label_B.sum() / 255
        # print(f"==>> sizeB: {sizeB}")
        
        # cv2.imwrite('A.png', patch_A[:, :, ::-1])
        # cv2.imwrite('B.png', patch_B[:, :, ::-1])
        # cv2.imwrite('A_label.png', label_A)
        # cv2.imwrite('B_label.png', label_B)
        
        sim = pair['sim']
        # print(f"==>> sim: {sim}")
        # print(f"==>> sim: {sim}")
        return self.transform(patch_A), self.transform(patch_B), torch.tensor(sim).float().unsqueeze(0)

    def __len__(self):
        return len(self.pair_list)

if __name__ == '__main__':
    dataset = Dataset(mode='train', dataset_dir_paths=['./DamDataset'])
    print(len(dataset))
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0)
    for a, b, label in loader:
        print(f"==>> label: {label}")
        print(f"==>> b.shape: {b.shape}")
        print(f"==>> a.shape: {a.shape}")
        