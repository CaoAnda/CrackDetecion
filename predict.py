import os
from utils.predict_utils import Predicter
from utils.tools import get_prompt


if __name__ == '__main__':
    # 13, 20
    image_path = "./DamDataset/dataV1/image/creak_s4.jpg"
    
    log_dir = "Nov08_08_30_07-the first try"
    
    # 预测单张 or 推理整个数据集
    single_predict = True
    conf_thr = 0.5
    # 0.9比较准确
    predictor = Predicter(
        log_path=f'logs/{log_dir}',
        conf_thr=conf_thr,
        device='cuda:0'
    )
    
    if single_predict:
        result_dir = 'result'
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        bboxes, _ = predictor.predict_path(path=image_path, prompt_bbox=get_prompt(image_path.split('/')[-1]), outpath=os.path.join(result_dir, 'result.png'))
        with open(os.path.join(result_dir, 'result.txt'), 'w+') as file:
            for box in bboxes:
                for i in box:
                    file.write(str(i)+' ')
                file.write('\n')
    else:
        image_dir_path = './DamDataset/image'

        result_root_path = f'./DamDataset/result'
        if not os.path.exists(result_root_path):
            os.mkdir(result_root_path)
        
        result_dir_path = os.path.join(result_root_path, log_dir + f'-score({conf_thr})-length(8)')
        if not os.path.exists(result_dir_path):
            os.mkdir(result_dir_path)
        
        result_image_dir_path = os.path.join(result_dir_path, 'image')
        if not os.path.exists(result_image_dir_path):
            os.mkdir(result_image_dir_path)
        
        result_box_dir_path = os.path.join(result_dir_path, 'box')
        if not os.path.exists(result_box_dir_path):
            os.mkdir(result_box_dir_path)
        
        for filename in os.listdir(image_dir_path):
            filepath = os.path.join(image_dir_path, filename)
            result_box_out_path = os.path.join(result_box_dir_path, filename[:-4]+'.txt')
            result_image_out_path = os.path.join(result_image_dir_path, filename[:-4]+'.jpg')
            bboxes, probs = predictor.predict_path(filepath, outpath=result_image_out_path)
            
            with open(result_box_out_path, 'w+') as file:
                for box, prob in zip(bboxes, probs):
                    for i in box:
                        file.write(str(i)+' ')
                    for i in prob:
                        file.write(str(i.item())+' ')
                    # file.write(str(prob.tolist()))
                    file.write('\n')