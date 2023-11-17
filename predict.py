import os
from utils.predict_utils import Predicter
from utils.tools import get_prompt
from tqdm import tqdm


if __name__ == '__main__':
    # 13, 20
    image_path = "./DamDataset/dataV1/image/creak_s34.jpg"
    box_root = './DamDataset/result/Jun02_06_33_42/box'
    # log_dir = "Nov08_08_30_07-the first try"
    
    # image_path = "./DamDataset/dataV2/image/creak_p8.jpg"
    # box_root = './DamDataset/dataV2/result/Jun02_06_33_42/box'
    
    # image_path = "DamDataset/Crack500/image/20160328_153559.jpg"
    # box_root = "DamDataset/Crack500/result/Jun02_06_33_42-filter(3)/box"
    # log_dir = "Nov09_06_22_24-Crack500-vipa-02"
    log_dir = "Nov10_06_29_33-Crack500+weight_decay(5e-3)-star"
    log_dir = "Nov16_13_03_45-Crack500+enhanced+cos_sim-vipa-209"
    
    
    
    # 预测单张 or 推理整个数据集
    single_predict = True
    conf_thr = 0.7
    # 0.9比较准确
    predictor = Predicter(
        log_path=f'logs/{log_dir}',
        conf_thr=conf_thr,
        device='cuda:0'
    )
    
    if single_predict:
        result_dir = 'result'
        prompt = [0, 0, 64, 64]
        # prompt = get_prompt(image_path.split('/')[-1], root=box_root)
        
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        bboxes, _ = predictor.predict_path(path=image_path, prompt_bbox=prompt, outpath=os.path.join(result_dir, 'result.png'))
        with open(os.path.join(result_dir, 'result.txt'), 'w+') as file:
            for box in bboxes:
                for i in box:
                    file.write(str(i)+' ')
                file.write('\n')
    else:
        image_dir_path = './DamDataset/Crack500/image'
        box_root = "DamDataset/Crack500/result/Jun02_06_33_42-filter(3)/box"
        result_root_path = f'./DamDataset/Crack500/result'
        if not os.path.exists(result_root_path):
            os.mkdir(result_root_path)
        
        result_dir_path = os.path.join(result_root_path, log_dir + f'-score({conf_thr})')
        if not os.path.exists(result_dir_path):
            os.mkdir(result_dir_path)
        
        result_image_dir_path = os.path.join(result_dir_path, 'image')
        if not os.path.exists(result_image_dir_path):
            os.mkdir(result_image_dir_path)
        
        result_box_dir_path = os.path.join(result_dir_path, 'box')
        if not os.path.exists(result_box_dir_path):
            os.mkdir(result_box_dir_path)
        
        for filename in tqdm(os.listdir(image_dir_path), ncols=100):
            filepath = os.path.join(image_dir_path, filename)
            result_box_out_path = os.path.join(result_box_dir_path, filename[:-4]+'.txt')
            result_image_out_path = os.path.join(result_image_dir_path, filename[:-4]+'.jpg')
            prompt = get_prompt(filepath.split('/')[-1], root=box_root)
            
            bboxes, probs = predictor.predict_path(path=filepath, prompt_bbox=prompt, outpath=result_image_out_path)
            
            with open(result_box_out_path, 'w+') as file:
                for box, prob in zip(bboxes, probs):
                    for i in box:
                        file.write(str(i)+' ')
                    file.write(str(prob)+' ')
                    file.write('\n')