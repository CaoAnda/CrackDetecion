import os
from queue import Queue
import torch
import yaml

import torchvision.transforms as transforms
import numpy as np
from PIL import Image, ImageDraw
from functools import partial
from tqdm import tqdm

from utils.tools import get_box, get_patch_num, imread

tqdm = partial(tqdm, ncols=100)
Image.MAX_IMAGE_PIXELS = 150000000


class Predicter():

    def __init__(self,
                 log_path: str,
                 conf_thr: float = 0.5,
                 connected_thr: float = 0.5,
                 use_sequence: bool = False,
                 repetition: int = 1,
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
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu')
        self.opt = self.config_init(log_path)
        self.model = self.model_init(log_path)
        self.seq_len = self.opt['seq_len']
        self.conf_thr = conf_thr
        self.connected_thr = connected_thr
        self.use_sequence = use_sequence
        self.repetition = repetition
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5990, 0.6012, 0.5874),
                                 std=(0.0383, 0.0380, 0.0374)),
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
        from model.net import Net
        model = Net(**self.opt).to(self.device)
        model.load_state_dict(
            torch.load(
                os.path.join(log_path, 'weights',
                             f'model_direction_f1score.pkl')))
        model.eval()
        return model

    def predict_patch(self, patch: np.ndarray, *args):
        """
        预测切片结果

        Parameters
        ----------
        patch : np.ndarray
            切片

        Returns
        -------
        _type_
            预测结果，（两个模型状态值）
        """
        input = self.transform(patch).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output, (h, c) = self.model(input, *args)
        return output[0], (h, c)

    def predict_patches(self, patches: list):
        """
        预测序列切片结果

        Parameters
        ----------
        patches : list[tensor]
            切片序列

        Returns
        -------
        _type_
            序列中最后一个切片的预测结果
        """
        input = torch.stack(patches).to(self.device)
        with torch.no_grad():
            output, (h, c) = self.model(input)
        return output[-1], (h, c)

    def boxes_append(self, boxes, xmin, ymin, xmax, ymax, h, w):
        xmin -= self.left
        xmax -= self.left
        ymin -= self.top
        ymax -= self.top
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(xmax, w)
        ymax = min(ymax, h)
        boxes.append([xmin, ymin, xmax, ymax])

    def predict_image(self, image: np.ndarray, image_h:int, image_w:int) -> tuple[list, list]:
        """预测图像

        Args:
            image (np.ndarray): 图像

        Returns:
            tuple[list, list]: 图像预测结果，格式为boxes[list], probs[list]
        """
        patch_num_h, patch_num_w = get_patch_num(image, self.opt['patch_size'])
        vis_flag = [[0] * patch_num_w for _ in range(patch_num_h)]
        # 将下面初始值置为True，就可以测试没有"二次预测"的对比效果，ps. 二次预测：如果一个patch前期被预测为不是裂缝，但是后续又被认为是裂缝，那么就对它进行再预测
        crack_flag = [[False] * patch_num_w for _ in range(patch_num_h)]

        boxes = []
        probs = []
        for y in tqdm(range(patch_num_h)):
            for x in range(patch_num_w):
                if vis_flag[y][x] == self.repetition:
                    continue
                vis_flag[y][x] += 1
                xmin, ymin, xmax, ymax = get_box(self.opt['patch_size'], x, y)
                patches = [self.transform(image[ymin:ymax, xmin:xmax])]
                prob, (h, c) = self.predict_patches(patches)

                if (prob >= self.conf_thr).any():
                    self.boxes_append(boxes, xmin, ymin, xmax, ymax, image_h, image_w)
                    probs.append(prob)
                    crack_flag[y][x] = True
                    queue = Queue()
                    node = dict(x=x, y=y, prob=prob, h=h, c=c, patches=patches)
                    if not self.use_sequence:
                        continue
                    queue.put(node)
                    while not queue.empty():
                        dx = [1, -1, 0, 0]
                        dy = [0, 0, 1, -1]
                        node = queue.get()
                        for i in range(4):
                            _x = node['x'] + dx[i]
                            _y = node['y'] + dy[i]

                            if node['prob'][
                                    i] < self.connected_thr or _x < 0 or _x >= patch_num_w or _y < 0 or _y >= patch_num_h or vis_flag[
                                        _y][_x] == self.repetition:
                                continue
                            vis_flag[_y][_x] += 1

                            _xmin, _ymin, _xmax, _ymax = get_box(
                                self.opt['patch_size'], _x, _y)
                            _patches = node['patches'].copy()
                            _patches.append(self.transform(image[_ymin:_ymax, _xmin:_xmax]))
                            if len(_patches) > self.seq_len:
                                _patches.pop(0)
                            _prob, (_h, _c) = self.predict_patches(_patches)

                            if (_prob >= self.conf_thr).any():
                                crack_flag[_y][_x] = True
                                # if (node['prob'][i] >= self.conf_thr).any():
                                _node = dict(x=_x,
                                             y=_y,
                                             prob=_prob,
                                             h=_h,
                                             c=_c,
                                             patches=_patches)
                                self.boxes_append(boxes, _xmin, _ymin, _xmax, _ymax, image_h, image_w)
                                probs.append(_node['prob'])
                                queue.put(_node)
        return boxes, probs

    def predict_path(self, path: str, outpath: str = ""):
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
        image, (top, left) = imread(path, -1, self.opt['patch_size'])
        _image, *_ = imread(path, -1, self.opt['patch_size'], expand=False)
        self.top = top
        self.left = left
        h, w, *_ = _image.shape
        boxes, probs = self.predict_image(image, image_h=h, image_w=w)
        if outpath:
            result_image = self.draw_boxes_on_image(_image,
                                                    boxes,
                                                    probs,
                                                    draw_prob=False)
            result_image.save(outpath)
        return boxes, probs

    # def draw_boxes_on_image(self,
    #                         image: np.ndarray,
    #                         boxes: np.ndarray,
    #                         probs: np.ndarray,
    #                         draw_prob: bool = False) -> Image:
    #     """
    #     给定PIL.Image对象和一个检测框列表，将检测框绘制在图像上，并返回绘制后的新图像。

    #     参数：
    #     image：PIL.Image对象，需要标注检测框的原始图像。
    #     boxes：检测框列表，每个检测框为一个四元组 (x_min, y_min, x_max, y_max)，
    #             分别表示检测框左上角和右下角的坐标值。
    #     probs: 检测框对应的概率值列表，每个概率值为一个浮点数。
    #     draw_prob: 是否在框的中心位置绘制对应的prob值。

    #     返回值：
    #     返回绘制后的新图像。
    #     """

    #     # 先将原图转换成可编辑的 ImageDraw 对象
    #     image_pil = Image.fromarray(np.uint8(image))
    #     draw = ImageDraw.Draw(image_pil)

    #     for box, prob in zip(boxes, probs):
    #         stroke_width = 3
    #         x_min, y_min, x_max, y_max = box
    #         draw.rectangle([x_min, y_min, x_max, y_max],
    #                        outline="red",
    #                        width=stroke_width)

    #         if draw_prob:
    #             prob_text = f"{prob:.2f}"
    #             text_width, text_height = draw.textsize(prob_text)
    #             text_x = x_min + stroke_width
    #             text_y = y_min + stroke_width
    #             draw.rectangle([
    #                 text_x, text_y, text_x + text_width, text_y + text_height
    #             ],
    #                            fill="red")
    #             draw.text((text_x, text_y), prob_text, fill="white")

    #     # 返回绘制后的新图像
    #     return image_pil
    def draw_boxes_on_image(self,
                            image: np.ndarray,
                            boxes: np.ndarray,
                            probs: list[list[float]],
                            draw_prob: bool = False) -> Image:
        """
        给定PIL.Image对象和一个检测框列表，将检测框绘制在图像上，并返回绘制后的新图像。
        
        参数：
        image：PIL.Image对象，需要标注检测框的原始图像。
        boxes：检测框列表，每个检测框为一个四元组 (x_min, y_min, x_max, y_max)，
                分别表示检测框左上角和右下角的坐标值。
        probs: 检测框对应的概率值列表，每个概率值为一个包含四个概率的列表，
            分别表示框向右、向左、向下、向上的概率。
        draw_prob: 是否在框的中心位置绘制对应的prob值。
        
        返回值：
        返回绘制后的新图像。
        """

        # 先将原图转换成可编辑的 ImageDraw 对象
        image_pil = Image.fromarray(np.uint8(image))
        draw = ImageDraw.Draw(image_pil)

        for box, box_probs in zip(boxes, probs):
            stroke_width = 3
            x_min, y_min, x_max, y_max = box

            # # 绘制框的右边
            # if box_probs[0] >= self.connected_thr:
            #     draw.line([(x_max, y_min), (x_max, y_max)], fill="red", width=stroke_width)

            # # 绘制框的左边
            # if box_probs[1] >= self.connected_thr:
            #     draw.line([(x_min, y_min), (x_min, y_max)], fill="red", width=stroke_width)

            # # 绘制框的下边
            # if box_probs[2] >= self.connected_thr:
            #     draw.line([(x_min, y_max), (x_max, y_max)], fill="red", width=stroke_width)

            # # 绘制框的上边
            # if box_probs[3] >= self.connected_thr:
            #     draw.line([(x_min, y_min), (x_max, y_min)], fill="red", width=stroke_width)

            arrow_length = 15
            # 绘制框的右边
            if box_probs[0] >= self.connected_thr:
                arrow_start = (x_max, (y_min + y_max) / 2)
                arrow_end = (x_max + arrow_length, (y_min + y_max) / 2)
                draw.line([arrow_start, arrow_end],
                          fill="red",
                          width=stroke_width)
                draw.polygon([
                    arrow_end, (arrow_end[0] - 7, arrow_end[1] - 4),
                    (arrow_end[0] - 7, arrow_end[1] + 4)
                ],
                             fill="red")

            # 绘制框的左边
            if box_probs[1] >= self.connected_thr:
                arrow_start = (x_min, (y_min + y_max) / 2)
                arrow_end = (x_min - arrow_length, (y_min + y_max) / 2)
                draw.line([arrow_start, arrow_end],
                          fill="red",
                          width=stroke_width)
                draw.polygon([
                    arrow_end, (arrow_end[0] + 7, arrow_end[1] - 4),
                    (arrow_end[0] + 7, arrow_end[1] + 4)
                ],
                             fill="red")

            # 绘制框的下边
            if box_probs[2] >= self.connected_thr:
                arrow_start = ((x_min + x_max) / 2, y_max)
                arrow_end = ((x_min + x_max) / 2, y_max + arrow_length)
                draw.line([arrow_start, arrow_end],
                          fill="red",
                          width=stroke_width)
                draw.polygon([
                    arrow_end, (arrow_end[0] - 4, arrow_end[1] - 7),
                    (arrow_end[0] + 4, arrow_end[1] - 7)
                ],
                             fill="red")

            # 绘制框的上边
            if box_probs[3] >= self.connected_thr:
                arrow_start = ((x_min + x_max) / 2, y_min)
                arrow_end = ((x_min + x_max) / 2, y_min - arrow_length)
                draw.line([arrow_start, arrow_end],
                          fill="red",
                          width=stroke_width)
                draw.polygon([
                    arrow_end, (arrow_end[0] - 4, arrow_end[1] + 7),
                    (arrow_end[0] + 4, arrow_end[1] + 7)
                ],
                             fill="red")

            draw.rectangle([x_min, y_min, x_max, y_max],
                           outline="red",
                           width=stroke_width)

            if draw_prob:
                prob_text = f"{box_probs}"
                text_width, text_height = draw.textsize(prob_text)
                text_x = x_min + stroke_width
                text_y = y_min + stroke_width
                draw.rectangle([
                    text_x, text_y, text_x + text_width, text_y + text_height
                ],
                               fill="red")
                draw.text((text_x, text_y), prob_text, fill="white")

        # 返回绘制后的新图像
        return image_pil