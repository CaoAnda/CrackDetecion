from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime

# 默认参数会在runs文件夹生成对应的日志文件夹，以及日志文件
writer = SummaryWriter()

# 指定日志文件夹目录
# now_time = datetime.now().strftime("%b%d_%H_%M_%S")
# log_dir = './logs/%s' % now_time
# writer = SummaryWriter(log_dir=log_dir)

for n_iter in range(100):
    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)