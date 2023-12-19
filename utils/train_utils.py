from datetime import datetime
from functools import partial
import time
import torch
import numpy as np
import os
from colorama import Fore
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from net.siamese import Net
from torch import nn
from utils.Dataloader import build_dataloader
from torch.utils.tensorboard import SummaryWriter
from utils.tools import squash_packed, WarmupCosineSchedule


def get_evalutaion_index(TP_FP, gross, TP, TP_TN, TP_FN, prefix=""):
    """
     计算acc, recall, precision, f1score

     @param correct - 预测正确的样本数量.
     @param total - 样本数量
     @param Positive_correct - 正样本中预测对的数量
     @param Positive_label_num - 样本中正样本的数量
     @param Positive_pred_num - 预测为正样本的数量

     @return acc, recall, precision, f1score
    """
    epsilon = 1e-8
    acc = TP_TN / (gross + epsilon)
    recall = TP.item() / (TP_FP.item() + epsilon)
    precision = TP.item() / (TP_FN.item() + epsilon)
    f1score = (1 + 1) * (recall * precision) / (recall + precision + epsilon)

    score = dict(
        acc=acc,
        recall=recall,
        precision=precision,
        f1score=f1score,
    )

    if prefix:
        keys = [key for key in score.keys()]
        for key in keys:
            score[prefix + '_' + key] = score.pop(key)
    return score


class Trainer():

    def __init__(self, device, optim: str, init_lr: float, min_lr: float,
                 epochs: int, score_thr: float, log_dir: str,
                 weight_decay: float, **kwargs) -> None:
        """用于训练模型

        Args:
            model (_type_): 要训练的模型
            device (_type_): 运行的设备
            criterion (_type_): loss函数
            optimizer (_type_): 优化器
            lr_scheduler (_type_): 用于学习率衰减
            train_loader (_type_): 用于加载训练集
            val_loader (_type_): 用于加载验证集
            tqdm (_type_): 用于加载进度条
            tb_writer (_type_): 用于在tensorboard中记录各项指标信息
            log_dir (str): 日志文件目录
        """
        local_params = locals()
        local_params.update(kwargs)
        self.model = Net(**kwargs).to(device)
        self.device = device
        self.epochs = epochs
        self.score_thr = score_thr
        # self.criterion = nn.MSELoss()
        self.criterion = nn.BCELoss()
        self.optimizer = getattr(torch.optim, optim)(self.model.parameters(),
                                                     lr=init_lr,
                                                     weight_decay=weight_decay)
        self.enhanced = kwargs['enhanced']
        self.train_loader = None if kwargs[
            'test4lr'] == 1 else build_dataloader(mode='train', **kwargs)
        self.val_loader = None if kwargs['test4lr'] == 2 else build_dataloader(
            mode='val', **kwargs)
        self.lr_scheduler = WarmupCosineSchedule(self.optimizer,
                                                 warmup_epochs=1,
                                                 total_epochs=epochs,
                                                 min_lr=min_lr)
        self.tqdm = partial(tqdm, ncols=100)
        self.log_dir = log_dir
        self.tb_writer = SummaryWriter(
            log_dir=os.path.join(self.log_dir, 'runs'))
        self.best_val_score = dict()
        self.iter_index = 0  # 因为每个epoch的iter个数可能不一样，所以需要用这个变量记录
        self.tb_writer.add_text('config', str(local_params))

    def run(self, epoch: int, loader, mode: str) -> dict:
        """完成一个epoch的推理工作

        Args:
            epoch (int): 当前epoch ID
            loader (_type_): 用于数据加载
            mode (str): 用于确定当前是训练还是验证

        Returns:
            dict: 返回模型当前各项指标
        """
        assert mode in ['train', 'val']
        device = self.device
        color = (Fore.GREEN if mode == 'val' else Fore.YELLOW)
        tqdm_bar = self.tqdm(loader, desc=color + f'{mode} epoch [{epoch}]')

        load_time_list = []
        loss_list = []
        
        last_time = time.time()
        gross = 0
        correct = 0
        iters = len(loader)
        for index, (A, labels) in enumerate(tqdm_bar):
            now_time = time.time()

            A = A.to(device)
            # B = B.to(device)
            labels = labels.to(device)
            # outputs = self.model(A, B)
            outputs = self.model(A)
            loss = self.criterion(outputs, labels)
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.tb_writer.add_scalar('train/lr',
                                          self.lr_scheduler.get_last_lr()[0],
                                          self.iter_index)
                self.iter_index += 1
                self.lr_scheduler.step(epoch + index / iters)

            loss_list.append(loss.item())

            end_time = time.time()
            load_time_list.append(now_time - last_time)
            tqdm_bar.set_postfix({
                'Load':
                f'{sum(load_time_list) if len(load_time_list)<10 else sum(load_time_list[-10:]):.2f} s',
                'Cal': f'{end_time-now_time:.2f} s'
            })
            last_time = time.time()

            preds = torch.where(outputs > self.score_thr, 1., 0.)
            gross += labels.numel()
            correct += (preds == labels).sum()

        loss = np.average(loss_list)
        
        result_score = {}
        result_score['loss'] = loss.item()
        result_score['acc'] = correct / gross
        
        print(
            color +
            '   '.join([f'{key}: {result_score[key]}'
                        for key in result_score]))
        return result_score

    def train_epoch(self, epoch: int) -> dict:
        """模型训练

        Args:
            epoch (int): 当前epoch ID

        Returns:
            dict: 当前模型在训练集上的各项指标表现
        """
        self.model.train()
        train_socre = self.run(epoch, self.train_loader, mode='train')

        return train_socre

    def val_epoch(self, epoch: int) -> dict:
        """模型验证

        Args:
            epoch (int): 当前epoch ID

        Returns:
            dict: 当前模型在验证集上的各项指标表现
        """
        self.model.eval()
        with torch.no_grad():
            val_score = self.run(epoch, self.val_loader, mode='val')
        return val_score

    def record_tensorboard(self,
                           epoch: int,
                           train_score: dict,
                           val_score: dict = dict()):
        """用于在tensorboard记录各项指标信息

        Args:
            epoch (int): 当前epoch ID
            train_score (dict): 当前模型在训练集上的各项指标表现
            val_score (dict): 当前模型在验证集上的各项指标表现
        """
        for mode in ['train', 'val']:
            score = eval(f'{mode}_score')
            for key in score:
                self.tb_writer.add_scalar(f'{mode}/{key}', score[key], epoch)

    def save_model(self, val_score: dict):
        """保存模型

        Args:
            val_score (dict): 当前模型在验证集上的各项指标表现
        """
        save_dir = os.path.join(self.log_dir, 'weights')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for key in val_score:
            if key not in self.best_val_score.keys(
            ) or val_score[key] > self.best_val_score[key]:
                self.best_val_score[key] = val_score[key]
                torch.save(self.model.state_dict(),
                           os.path.join(save_dir, f'model_{key}.pkl'))
        torch.save(self.model.state_dict(),
                           os.path.join(save_dir, f'model_last.pkl'))

    def run_train_val_epoch(self, epoch: int):
        """模型完成一个epoch的训练和验证, 记录日志并保存最优权重

        Args:
            epoch (int): 当前epoch ID
        """
        train_score = self.train_epoch(epoch)
        val_score = self.val_epoch(epoch)
        self.record_tensorboard(epoch, train_score, val_score)
        self.save_model(val_score)
        print()

    def enhance_images(self, epoch: int, test4lr: int=0):
        start_time = time.time()
        train_data = self.train_loader.dataset
        while train_data.enhanced_thread.is_alive():
            print(
                Fore.LIGHTBLACK_EX +
                f'\rLoading Enhanced Images: {train_data.get_enhanced_process(epoch)}, Time: {time.time()-start_time:.2f} s. ',
                end='')
            time.sleep(2)
        else:
            print(
                Fore.BLUE +
                f'Finish Loading Enhanced Images {train_data.get_enhanced_process(epoch)}.'
            )
        train_data.enhanced_thread.join()
        train_data.access_pre_setting(epoch)
        if epoch + 1 < self.epochs and not test4lr:
            train_data.start_enhanced_thread(epoch + 1)

    def train(self):
        for epoch in range(self.epochs):
            if self.enhanced:
                self.enhance_images(epoch)
            self.run_train_val_epoch(epoch)
        self.tb_writer.close()

    def Test4lr(self, optim: str, batch_size: int, weight_decay: float,
                test4lr: int, **kargs):
        """用于找到最优学习率
        """
        init_lr = 1e-7
        end_lr = 0.01
        self.optimizer = getattr(torch.optim, optim)(self.model.parameters(),
                                                     lr=init_lr,
                                                     weight_decay=weight_decay)
        loader = self.val_loader if test4lr == 1 else self.train_loader
        if test4lr == 2:
            self.enhance_images(0, test4lr=test4lr)
        step_count = min(len(loader), 40)
        loader.dataset.sequences_list = loader.dataset.sequences_list[:
                                                                      step_count
                                                                      *
                                                                      batch_size]
        lr_scheduler = StepLR(self.optimizer,
                              step_size=1,
                              gamma=(end_lr / init_lr)**(1 / step_count))

        self.model.train()
        device = self.device
        color = Fore.YELLOW
        tqdm_bar = self.tqdm(loader, desc=color + 'Test For lr')

        for index, (seqs, labels) in enumerate(tqdm_bar):
            seqs = seqs.to(device)
            labels = labels.to(device)

            outputs = self.model(seqs)

            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            preds = torch.where(outputs > self.score_thr, 1., 0.)
            gross = labels.numel()
            TP_FP = labels.sum()
            TP_FN = preds.sum()
            TP = torch.sum(torch.logical_and(labels == 1, preds == 1))
            TP_TN = (preds == labels).sum().item()

            result_score = get_evalutaion_index(TP_FP,
                                                gross,
                                                TP,
                                                TP_TN,
                                                TP_FN,
                                                prefix='direction')
            result_score['loss'] = loss.item()

            self.record_tensorboard(index, result_score)
            self.tb_writer.add_scalar('Test4lr/lr',
                                      lr_scheduler.get_last_lr()[0], index)
            lr_scheduler.step()
