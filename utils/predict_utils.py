import os
from queue import Queue
import torch
import yaml

import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
from functools import partial
from tqdm import tqdm

from utils.tools import get_box, get_patch_num, imread, get_patch
from utils.Dataset import Dataset

tqdm = partial(tqdm, ncols=100)
Image.MAX_IMAGE_PIXELS = 150000000


class Predicter():

    def __init__(self,
                 log_path: str,
                 conf_thr: float = 0.5,
                 device: str = 'cuda') -> None:
        """
        Predicter初始化

        Parameters
        ----------
        log_path : str
            日志路径
        conf_thr : float, optional
            置信度阈值, by default 0.5
        connected_thr : float, optional
            判断连通的阈值, by default 0.5
        """
        self.__dict__.update(self.config_init(log_path))
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.model = self.model_init(log_path)
        self.conf_thr = conf_thr
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.6121, 0.6286, 0.6258),
                                 std=(0.0358, 0.0367, 0.0370)),
        ])

    def config_init(self, log_path: str):
        with open(os.path.join(log_path, 'config.yml')) as f:
            opt = yaml.load(f, Loader=yaml.FullLoader)
        return opt

    def model_init(self, log_path: str):
        """
        模型初始化

        Parameters
        ----------
        log_path : str
            日志路径

        Returns
        -------
        _type_
            模型对象
        """
        from net.siamese import Net
        model = Net(**self.__dict__).to(self.device)
        model.load_state_dict(
            torch.load(
                os.path.join(log_path, 'weights',
                             f'model_acc.pkl'), map_location=self.device))
        model.eval()
        return model

    def boxes_update(self, xmin, ymin, xmax, ymax, top, left, h, w):
        xmin -= left
        xmax -= left
        ymin -= top
        ymax -= top
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax, w)
        ymax = min(ymax, h)
        return [xmin, ymin, xmax, ymax]

    def predict_image(self, expand_image: np.ndarray, prompt_bbox: list, top: int, left: int) -> tuple[list, list]:        
        """
        预测图像中存在裂缝的box坐标以及其置信度

        Parameters
        ----------
        image : np.ndarray
            扩展后的图像

        Returns
        -------
        tuple[list, list]
            图像预测结果，格式为boxes[list], probs[list]
        """
        boxes = []
        probs = []
        
        patch_h, patch_w = get_patch_num(expand_image, self.patch_size)
        xmin, ymin, xmax, ymax = prompt_bbox
        xmin += left
        xmax += left
        ymin += top
        ymax += top
        
        
        # self.draw_boxes_on_image(expand_image, [], [], [xmin, ymin, xmax, ymax]).save('tmp.png')
        batch_size = 256
        batch_indexes_list = [
            [x, y] for y in range(patch_h) for x in range(patch_w)
        ]
        list_len = len(batch_indexes_list)
        patch_prompt = self.transform(expand_image[ymin:ymax, xmin:xmax]).unsqueeze(0)
        
        for s in range(0, list_len, batch_size):
            batchA = []
            batchB = []
            # print(f"==>> batch_indexes_list[s:min(list_len, s + batch_size)]: {batch_indexes_list[s:min(list_len, s + batch_size)]}")
            
            for x, y in batch_indexes_list[s:min(list_len, s + batch_size)]:
                patch_pred = self.transform(get_patch(expand_image, x, y, self.patch_size)).unsqueeze(0)
                batchA.append(patch_prompt)
                batchB.append(patch_pred)
            batchA = torch.cat(batchA, dim=0).to(self.device)
            batchB = torch.cat(batchB, dim=0).to(self.device)
            with torch.no_grad():
                similarities = self.model(batchA, batchB)
                # print(f"==>> similarities: {similarities}")
            for index in range(len(similarities)):
                x, y = batch_indexes_list[s:min(list_len, s + batch_size)][index]
                if similarities[index] > self.conf_thr:
                    boxes.append(get_box(self.patch_size, x, y))
                    probs.append(similarities[index].item())

        return boxes, probs

    def predict_path(self, path: str, prompt_bbox: list, outpath: str = ""):
        """
        根据图像路径，获取对应图像预测结果

        Parameters
        ----------
        path : str
            图像路径
        outpath : str, optional
            保存结果图像的路径, by default ""

        Returns
        -------
        _type_
            返回预测结果，坐标list和置信度list
        """
        image, (top, left, h, w) = imread(path, -1, self.patch_size)
        _image, *_ = imread(path, -1, self.patch_size, expand=False)
        
        boxes, probs = self.predict_image(image, prompt_bbox, top, left)
        boxes = [self.boxes_update(*box, top, left, h, w) for box in boxes]
        # boxes, probs = self.filter_connected_boxes(boxes, probs, n_limit=3)
        
        if outpath:
            result_image = self.draw_boxes_on_image(_image,
                                                    boxes,
                                                    probs,
                                                    prompt_bbox=prompt_bbox,
                                                    draw_prob=True)
            result_image.save(outpath)
        return boxes, probs

    def draw_box_on_image(self, draw, box, prob, color='red'):
        stroke_width = 3
        x_min, y_min, x_max, y_max = box
        draw.rectangle([x_min, y_min, x_max, y_max],
                        outline=color,
                        width=stroke_width)

        prob_text = f"{prob:.2f}"
        text_width, text_height = draw.textsize(prob_text)
        text_x = x_min + stroke_width
        text_y = y_min + stroke_width
        draw.rectangle([
            text_x, text_y, text_x + text_width, text_y + text_height
        ], fill=color)
        draw.text((text_x, text_y), prob_text, fill="white")
    
    def draw_boxes_on_image(self,
                            image: np.ndarray,
                            boxes: np.ndarray,
                            probs: np.ndarray,
                            prompt_bbox: list = None,
                            draw_prob: bool = False) -> Image:
        """
        给定PIL.Image对象和一个检测框列表，将检测框绘制在图像上，并返回绘制后的新图像。

        参数：
        image：PIL.Image对象，需要标注检测框的原始图像。
        boxes：检测框列表，每个检测框为一个四元组 (x_min, y_min, x_max, y_max)，
                分别表示检测框左上角和右下角的坐标值。
        probs: 检测框对应的概率值列表，每个概率值为一个浮点数。
        draw_prob: 是否在框的中心位置绘制对应的prob值。

        返回值：
        返回绘制后的新图像。
        """

        # 先将原图转换成可编辑的 ImageDraw 对象
        image_pil = Image.fromarray(np.uint8(image))
        draw = ImageDraw.Draw(image_pil)

        for box, prob in zip(boxes, probs):
            self.draw_box_on_image(draw, box, prob, 'red')
        
        if prompt_bbox:
            self.draw_box_on_image(draw, prompt_bbox, 1, color='green')

        # 返回绘制后的新图像
        return image_pil

    def filter_connected_boxes(self, crack_boxes, crack_probs, n_limit):
        connected_blocks = []
        visited_boxes = [False] * len(crack_boxes)

        # 定义一个函数，用于深度优先搜索遍历连通块
        def dfs(box_index, connected_block):
            visited_boxes[box_index] = True
            connected_block.append(box_index)
            xmin, ymin, xmax, ymax = crack_boxes[box_index]

            # 遍历其他未访问过的检测框，如果相邻则加入连通块
            for i in range(len(crack_boxes)):
                if not visited_boxes[i]:
                    x_min, y_min, x_max, y_max = crack_boxes[i]
                    if x_min <= xmax and x_max >= xmin and y_min <= ymax and y_max >= ymin:
                        dfs(i, connected_block)

        # 遍历所有的检测框，构建连通块
        for i in range(len(crack_boxes)):
            if not visited_boxes[i]:
                connected_block = []
                dfs(i, connected_block)
                connected_blocks.append(connected_block)

        # 筛除检测框个数少于等于 n_limit 的连通块
        filtered_blocks = []
        for block in connected_blocks:
            if len(block) > n_limit:
                filtered_blocks.append(block)

        # 构建筛选后的检测框和置信度列表
        filtered_boxes = [crack_boxes[i] for block in filtered_blocks for i in block]
        filtered_probs = [crack_probs[i] for block in filtered_blocks for i in block]

        return filtered_boxes, filtered_probs