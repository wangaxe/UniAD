import sys
sys.path.append('/home/fu/workspace/UniAD')
# print(sys.path)

import argparse
import cv2
import torch
import sklearn
import mmcv
import os
import warnings

import importlib

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet3d.apis import single_gpu_test
from mmdet3d.models import (Base3DDetector, Base3DSegmentor,
                            SingleStageMono3DDetector)
from mmcv.image import tensor2imgs
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.uniad.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import time
import os.path as osp
from pprint import pprint
import torch.distributed as dist

from torchvision.utils import save_image


# dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
warnings.filterwarnings("ignore")

def get_img_tensor(mmcv_data:dict) -> torch.Tensor:
    return mmcv_data['img'][0].data[0]

def normlise_img(img_tensor:torch.Tensor) -> torch.Tensor:
    return (img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min())

def save_images(img_tensor:torch.Tensor) -> None:
    assert img_tensor.shape[1] == 6
    # for i in range()
    pass

def get_args(arg_lst):
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='output/results.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args(arg_lst)
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

arg_lst = [
    '/home/fu/workspace/UniAD/projects/configs/stage1_track_map/base_track_map.py',
    '/home/fu/workspace/UniAD/ckpts/uniad_base_e2e.pth',
    '--eval', 'bbox',
    '--show-dir', './projects/work_dirs/stage1_track_map/base_track_map/',
]
args = get_args(arg_lst)
cfg = Config.fromfile(args.config)
# pprint(dict(cfg))

plugin_dir = cfg.plugin_dir
_module_dir = os.path.dirname(plugin_dir)
_module_dir = _module_dir.split('/')
_module_path = _module_dir[0]

for m in _module_dir[1:]:
    _module_path = _module_path + '.' + m
plg_lib = importlib.import_module(_module_path)

cfg.data.test.test_mode = True
samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
if samples_per_gpu > 1:
    cfg.data.test.pipeline = replace_ImageToTensor(
        cfg.data.test.pipeline)
data_arg = cfg.data.test.copy()
data_arg.pop('type')
dataset = NuScenesE2EDataset(**data_arg)

data_loader = build_dataloader(
    dataset,
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=cfg.data.workers_per_gpu,
    dist=False,
    shuffle=False,
    nonshuffler_sampler=cfg.data.nonshuffler_sampler,
)
# print(samples_per_gpu)
# print(cfg.data.workers_per_gpu)
# print(cfg.data.nonshuffler_sampler)


# build the model and load checkpoint
cfg.model.train_cfg = None
print(cfg.model['seg_head'])
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
# fp16_cfg = cfg.get('fp16', None)
# if fp16_cfg is not None:
#     print('fp16')
#     wrap_fp16_model(model)
checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
# if args.fuse_conv_bn:
#     print('fuse_bn')
#     model = fuse_conv_bn(model)
# old versions did not save class info in checkpoints, this walkaround is
# for backward compatibility
if 'CLASSES' in checkpoint.get('meta', {}):
    model.CLASSES = checkpoint['meta']['CLASSES']
else:
    model.CLASSES = dataset.CLASSES
# palette for visualization in segmentation tasks
if 'PALETTE' in checkpoint.get('meta', {}):
    model.PALETTE = checkpoint['meta']['PALETTE']
elif hasattr(dataset, 'PALETTE'):
    # segmentation dataset has `PALETTE` attribute
    model.PALETTE = dataset.PALETTE

model = MMDataParallel(model, device_ids=[0])
model.eval()
# print(args.show) # False
# print(args.show_dir) # ./projects/work_dirs/stage1_track_map/base_track_map/
# print(type(dataset)) # <class 'projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset.NuScenesE2EDataset'>


# print(dir(dataset))
# print()
# print(dir(data_loader))

print(dataset.test_mode)
print(data_arg['pipeline'])
tmp_data = next(data_loader.__iter__())
print(tmp_data.keys())
print()

img_tensor = get_img_tensor(tmp_data)
# print(img_tensor.shape) # img_tensor.shape = [1, 6, 3, 928, 1600]
# print(img_tensor.squeeze()[1].shape)

print(img_tensor.squeeze()[1].min())
print(img_tensor.squeeze()[1].max())
img_tensor_1 = normlise_img(img_tensor.squeeze()[1])
print(img_tensor_1.max())
print(img_tensor_1.min())
save_image(img_tensor.squeeze()[1], './verification/test.png')
save_image(img_tensor_1, './verification/test_1.png')

with torch.no_grad():
    result = model(return_loss=False, rescale=True, **tmp_data)
#     # resutl is a Python dict
print(result[0].keys())
    # print(result[0]['scores_3d'].shape)

# outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)