import os
import torch
from os.path import join as opj
from argparse import ArgumentParser
import sys
from tqdm import tqdm


def swa(model_paths,best_n_model,save_dir,ckpt_dir,zipfile):
    model_num = len(model_paths)
    print('models nums: ', model_num)

    if model_num < best_n_model:
        model_paths = model_paths
        print(f'Loading {model_num} models...')
    else:
        model_paths = model_paths[(model_num - best_n_model):]
        print(f'Loading {best_n_model} models...')
    model_names = [i.split('/')[-1] for i in model_paths]
    print(f'load model {model_names}')

    models = [torch.load(model_path, map_location='cpu') for model_path in model_paths]
    model_num = len(models)
    model_keys = models[-1]['state_dict'].keys()
    state_dict = models[-1]['state_dict']
    new_state_dict = state_dict.copy()
    ref_model = models[-1]

    for key in tqdm(model_keys):
        sum_weight = 0.0
        for m in models:
            sum_weight += m['state_dict'][key]
        avg_weight = sum_weight / model_num
        new_state_dict[key] = avg_weight

    ref_model['state_dict'] = new_state_dict
    save_model_name = 'swa_best_' + str(best_n_model) + '.pth'
    if save_dir is not None:
        save_dir = os.path.join(save_dir, save_model_name)
    else:
        save_dir = os.path.join(ckpt_dir, save_model_name)

    torch.save(ref_model, save_dir, _use_new_zipfile_serialization=zipfile)
    print('Model is saved at', save_dir)

def swabyindex(ckpt_dir, indexs,best_n_model=10, save_dir=None, zipfile=False):
    # 文件后缀筛选规则
    if isinstance(indexs[0],int):
        model_paths = [opj(ckpt_dir, 'epoch_%d.pth'%(i)) for i in indexs]
    elif isinstance(indexs[0],float):
        model_paths = [opj(ckpt_dir, '%d_model.pth'%(i*10000)) for i in indexs]
    elif isinstance(indexs[0],str):
        model_paths = indexs
    else:
        raise NotImplemented
    swa(model_paths,best_n_model,save_dir,ckpt_dir,zipfile)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--ckpt_dir', default=
        '/home/hanglijun/xp/code/mmdetection/work_dirs/swinb_3x_800-1400_anchor_bs2x8_anchor_aug_iou_ad_2048'
        , help='the directory where checkpoints are saved')
    parser.add_argument('--best_n_model', default=5, help='select best n models for ensemble')
    parser.add_argument('--indexs', default=[21, 22, 23, 24, 25], help='select best n models for ensemble')
    parser.add_argument('--save_dir', default=None, help='the directory for saving the SWA model')
    args = parser.parse_args()

    swabyindex(args.ckpt_dir, args.indexs, args.best_n_model, save_dir=args.save_dir, zipfile=False)
