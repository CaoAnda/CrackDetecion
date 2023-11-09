import argparse  
  
def get_args(parser=argparse.ArgumentParser()):
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patch_size', type=int, default=64)
    # resnet18, resnet34, resnet50, swin_t, swin_s...
    parser.add_argument('--backbone_model', type=str, default='resnet18') #
    # 0, 1
    parser.add_argument('--backbone_pretrained', type=int, default=1) #
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--optim', type=str, default='Adam') #
    parser.add_argument('--weight_decay', type=float, default=0) #
    parser.add_argument('--init_lr', type=float, default=1e-4) #
    parser.add_argument('--min_lr', type=float, default=1e-6) #
    parser.add_argument('--seed', type=int, default=314)
    # 0, 1(spacial), 2(pixel), 3(spacial+pixel)
    parser.add_argument('--enhanced', type=int, default=0) #
    parser.add_argument('--score_thr', type=float, default=0.5) #
    parser.add_argument('--dataset_dir_paths', nargs='+', type=str, default=['./DamDataset/dataV1'])
    parser.add_argument('--test4lr', type=int, default=0) #
    parser.add_argument('--desc', type=str, required=True)
    parser.add_argument('--output', action='store_true', default=True)  
  
    opt = parser.parse_args()  
    if opt.output:
        print(opt)
    return opt

if __name__ == '__main__':
    opt = get_args()