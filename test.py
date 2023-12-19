import cv2
import os
from utils.predict_utils import Predicter
from utils.tools import get_dataset_pathes, imread, judge_crack, cal_crack_size, get_patch_num
from tqdm import tqdm

def get_POSITIVE_num(label_path:str, patch_size:int):
    """
    获得label标注中裂缝patch的个数

    Parameters
    ----------
    label_path : str
        label标注图像路径
    patch_size : int
        patch的大小

    Returns
    -------
    int
        裂缝patch的个数
    """
    label, _ = imread(label_path, flags=0, patch_size=patch_size)
    _, label = cv2.threshold(label, 10, 255, cv2.THRESH_BINARY)
    # print(f"==>> label.shape: {label.shape}")
    h, w = get_patch_num(label, patch_size)
    
    num = 0
    for y in range(h):
        for x in range(w):
            num += judge_crack(label, x, y, patch_size)
    
    return num

def get_TRUE_POSITIVE_num(label_path:str, bboxes:list, patch_size:int):
    """
    获得正阳性的个数

    Parameters
    ----------
    label_path : str
        label标注图像路径
    bboxes : list
        预测结果
    patch_size : int
        patch的大小

    Returns
    -------
    int
        正阳性裂缝patch的个数
    """
    label, _ = imread(label_path, flags=0, patch_size=patch_size, expand=False)
    _, label = cv2.threshold(label, 10, 255, cv2.THRESH_BINARY)
    
    num = 0
    for box in bboxes:
        num += (cal_crack_size(label, *box) >= 40)
    
    return num
    
def get_metrics(TP:int, FP:int, TN:int, FN:int):
    acc =  (TP + TN) / (TP + FP + TN + FN)
    precision =  TP / (TP + FP)
    recall = TP / (TP + FN)
    f1score = (1 + 1) * (recall * precision) / (recall + precision)
    return locals()
    
def test(dataset_dir_paths:str, predictor):
    image_paths, label_paths = get_dataset_pathes(dataset_dir_paths, mode='val')
    result_dir = 'result'
    prompt = [0, 0, 64, 64]
    
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    
    for image_path, label_path in tqdm(zip(image_paths, label_paths), ncols=100):
        # print(f"==>> label_path: {label_path}")
        
        bboxes, _ = predictor.predict_path(path=image_path, prompt_bbox=prompt, outpath=os.path.join(result_dir, 'result.png'))
        patch_size = predictor.patch_size
        label, _ = imread(label_path, flags=0, patch_size=patch_size, expand=False)
        h, w = get_patch_num(label, patch_size)
        
        POSITIVE = get_POSITIVE_num(label_path, patch_size=patch_size)
        TRUE_POSITIVE = get_TRUE_POSITIVE_num(label_path, bboxes, patch_size)
        FALSE_NEGATIVE = POSITIVE - TRUE_POSITIVE
        FALSE_POSITIVE = len(bboxes) - TRUE_POSITIVE
        TRUE_NEGATIVE = h * w - TRUE_POSITIVE - FALSE_NEGATIVE - FALSE_POSITIVE
        
        TP += TRUE_POSITIVE
        FP += FALSE_POSITIVE
        TN += TRUE_NEGATIVE
        FN += FALSE_NEGATIVE
        
    print(get_metrics(TP, FP, TN, FN))

if __name__ == '__main__':
    log_dir = "Dec05_10_35_03-True-VIPA207"
    dataset_dir_paths = 'DamDataset/dataV1/dataV1'
    score = 0.7
    connected_thr = 0.5
    predictor = Predicter(
        log_path=f'logs/{log_dir}',
        conf_thr=score,
        # connected_thr=connected_thr,
        # use_sequence=True,
        # repetition=1,
        device='cuda:1',
        # mode='direction_acc'
    )

    test(dataset_dir_paths, predictor)
    