import sys
sys.path.append('/home/fu/workspace/UniAD')
# print(sys.path)

import argparse
import cv2
import torch
import numpy as np
import sklearn
import mmcv
import os
import warnings
import copy

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
from mmdet3d.datasets import build_dataset, NuScenesDataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset import NuScenesE2EDataset
from mmdet3d.models import build_model
from mmdet.apis import set_random_seed
from projects.mmdet3d_plugin.uniad.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor

from projects.mmdet3d_plugin.datasets.data_utils.data_utils import output_to_nusc_box_det, lidar_nusc_box_to_global
from projects.mmdet3d_plugin.datasets.eval_utils.nuscenes_eval import DetectionBox_modified, load_gt

from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.loaders import load_prediction, add_center_dist, filter_eval_boxes
from nuscenes.eval.detection.data_classes import DetectionConfig, DetectionMetrics, DetectionBox, DetectionMetricDataList
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.common.utils import center_distance

from kornia.enhance import (adjust_brightness, adjust_gamma,
                            adjust_hue, adjust_saturation)
from projects.mmdet3d_plugin.verification import LowBoundedDIRECT_full_parrallel

import time
import os.path as osp
from pprint import pprint
import torch.distributed as dist

from torchvision.utils import save_image


# dist.init_process_group(backend='nccl', init_method='tcp://localhost:23456', rank=0, world_size=1)
warnings.filterwarnings("ignore")





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

    # perturbation
    parser.add_argument('--hue', default=0.01, type=float,
        help='range should be in [-PI, PI], while 0 means no shift'
        'control: [ - defautl * PI, defautl * PI]')
    parser.add_argument('--saturation', default=0.05, type=float,
        help='range should be in [0, 2], while 1 means no shift'
        'control: [ 1 - defautl, 1 + defautl]')
    parser.add_argument('--contrast', default=0.0, type=float,
        help='range should be in [0, 2], while 1 means no shift'
        'control: [ 1 - defautl, 1 + defautl]')
    parser.add_argument('--bright', default=0.1, type=float,
        help='range should be in [0, 1], while 0 means no shift'
        'control: TBD')
    # DIRECT
    parser.add_argument('--max-evaluation', default=5000, type=int)
    parser.add_argument('--max-deep', default=20, type=int)
    parser.add_argument('--po-set', action='store_true')
    parser.add_argument('--po-set-size', default=2, type=int)
    parser.add_argument('--max-iteration', default=50, type=int)
    parser.add_argument('--tolerance', default=1e-4, type=float)
    
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
# print(data_arg)
# data_arg['ann_file'] = '/home/fu/workspace/UniAD/data/infos/nuscenes_infos_temporal_val.pkl'
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
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))

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

####################################################
#            
####################################################

# print(args.show) # False
# print(args.show_dir) # ./projects/work_dirs/stage1_track_map/base_track_map/
# print(type(dataset)) # <class 'projects.mmdet3d_plugin.datasets.nuscenes_e2e_dataset.NuScenesE2EDataset'>

# print(dataset.test_mode)
# print(data_arg['pipeline'])
# print(tmp_data.keys())
# print(tmp_data['img_metas'][0])


def get_pred_boxes_single_frame(nusc_box,
                                sample_token,
                                mapped_class_names):
    nusc_annos = {}
    annos = []
    for i, box in enumerate(nusc_box):
        name = mapped_class_names[box.label]
        if np.sqrt(box.velocity[0]**2 + box.velocity[1]**2) > 0.2:
            if name in [
                    'car',
                    'construction_vehicle',
                    'bus',
                    'truck',
                    'trailer',
            ]:
                attr = 'vehicle.moving'
            elif name in ['bicycle', 'motorcycle']:
                attr = 'cycle.with_rider'
            else:
                attr = NuScenesDataset.DefaultAttribute[name]
        else:
            if name in ['pedestrian']:
                attr = 'pedestrian.standing'
            elif name in ['bus']:
                attr = 'vehicle.stopped'
            else:
                attr = NuScenesDataset.DefaultAttribute[name]

        nusc_anno = dict(
            sample_token=sample_token,
            translation=box.center.tolist(),
            size=box.wlh.tolist(),
            rotation=box.orientation.elements.tolist(),
            velocity=box.velocity[:2].tolist(),
            detection_name=name,
            detection_score=box.score,
            attribute_name=attr,
        )
        annos.append(nusc_anno)
        nusc_annos[sample_token] = annos
    return EvalBoxes.deserialize(nusc_annos, DetectionBox)

def process_nusc_boxes(boxes, dataset=dataset):
    class_range = dataset.eval_detection_configs.class_range 

    boxes = add_center_dist(dataset.nusc, boxes)
    boxes = filter_eval_boxes(dataset.nusc, boxes, class_range, verbose=False)
    return boxes

def results2dist(result: dict,
                 gt_boxes,
                 class_names,
                 dataset = dataset,
                 verbose=False
                 ):
    mapped_class_names = dataset.CLASSES
    sample_token = result['token']
    mean_dist = 0
    nb_positive_cls = 0

    boxes = output_to_nusc_box_det(result)
    boxes, keep_idx = lidar_nusc_box_to_global(
                                dataset.data_infos[0], boxes,
                                mapped_class_names,
                                dataset.eval_detection_configs,
                                dataset.eval_version)
    pred_boxes = get_pred_boxes_single_frame(boxes, 
                                            sample_token, 
                                                mapped_class_names)
    pred_boxes = process_nusc_boxes(pred_boxes)

    for class_name in class_names:
        class_distance = 0
        for pred_box in pred_boxes.all:
            if pred_box.detection_name != class_name: continue
            for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):
                if gt_box.detection_name == class_name:
                    class_distance += center_distance(gt_box, pred_box)
        if class_distance > 0:
            nb_positive_cls += 1
            mean_dist += class_distance
            # if verbose: print(class_name,' - ', class_distance)
    if nb_positive_cls != 0:
        return mean_dist/nb_positive_cls
    else: return 0

def negative_dist(result: dict,
                 gt_boxes,
                 class_names,
                 dataset = dataset,
                 verbose=False
                 ):
    return -1 * results2dist(result, gt_boxes, class_names,
                             dataset = dataset, verbose=False)

def get_gt_boxes(eval_set='mini_val', dataset=dataset, verbose=False):
    class_range = dataset.eval_detection_configs.class_range 
    gt_boxes = load_gt(dataset.nusc, 'mini_val', DetectionBox_modified, verbose=False)
    gt_boxes = process_nusc_boxes(gt_boxes)
    return gt_boxes

class FunctionalVerification():

    def __init__(self, 
                 model, 
                 status, # UniAD.module
                 frame:dict, 
                 ground_truth:tuple,
                 verify_loss): 
        self.model = model
        self.ori_frame = frame
        self.images = self.get_img_tensor(frame)
        self.loss = verify_loss
        self.gt, self.gt_cls = ground_truth
        self.nb_camera = self.images.shape[0]
        self.status = status

    def update_frame(self, new_frame):
        self.ori_frame = new_frame
        self.images = self.get_img_tensor(new_frame)
    
    @staticmethod
    def _normlise_img(img_tensor:torch.Tensor) -> torch.Tensor:
        return ((img_tensor - img_tensor.min()) / (img_tensor.max() - img_tensor.min()), (img_tensor.min(), img_tensor.max()))

    @staticmethod
    def _unnormlise_img(img_tensor:torch.Tensor, min_v, max_v) -> torch.Tensor:
        return img_tensor * (max_v - min_v) + min_v


    @staticmethod
    def get_img_tensor(mmcv_data:dict) -> torch.Tensor:
        return mmcv_data['img'][0].data[0].squeeze()

    @staticmethod
    def replace_img_tensor(ori_frame, perturb_imgs:torch.Tensor) -> dict:
        new_frame = copy.deepcopy(ori_frame)
        new_frame['img'][0].data[0] = perturb_imgs
        return new_frame

    @staticmethod
    def functional_perturb(x, in_arr):
        raise NotImplementedError

    def verification(self, in_arrs):
        query_result = []
        for idx, in_arr in enumerate(in_arrs):
            self.model.module = copy.deepcopy(self.status)
            perturbed_imgs = torch.zeros_like(self.images)
            in_arr = in_arr.reshape(self.nb_camera,-1)
            for i in range(self.nb_camera):
                perturbed_imgs[i] = self.functional_perturb(self.images[i], in_arr[i])
            tmp_frame = self.replace_img_tensor(self.ori_frame, perturbed_imgs.unsqueeze(0))
            with torch.no_grad():
                _pediction = self.model(return_loss=False, rescale=True, **tmp_frame)
            _query = self.loss(_pediction[0],
                                self.gt, self.gt_cls, verbose=True)
            query_result.append(_query)
        return np.array(query_result)

    def set_problem(self):
        return self.verification

class OpticalVerification(FunctionalVerification):
    @staticmethod
    def functional_perturb(x, in_arr):
        transformed = torch.zeros_like(x)
        # hue, saturation, contrast, bright = \
        #     in_arr[0], in_arr[1], in_arr[2], in_arr[3]
        # hue, saturation, contrast = in_arr[0], in_arr[1], in_arr[2]
        x, min_max_v = OpticalVerification._normlise_img(x)
        hue, saturation = in_arr[0], in_arr[1]
        with torch.no_grad():
            x = adjust_hue(x, hue)
            x = adjust_saturation(x, saturation)
            # x = adjust_gamma(x, contrast)
            # x = adjust_brightness(x, bright)
        x = OpticalVerification._unnormlise_img(x, *min_max_v)
        transformed = x
        return transformed 

bound = []
if args.hue != 0:
    bound.append([-np.pi*args.hue, np.pi*args.hue])
if args.saturation != 0:
    bound.append([1-args.saturation, 1+args.saturation])
if args.contrast != 0:
    bound.append([1-args.contrast, 1+args.contrast])
assert len(bound) != 0
bound = bound*6
print(bound)
####################################################
#            
####################################################


gt_boxes = get_gt_boxes()
simplied_class_names = ['car', 'truck', 'bus', 
                        'pedestrian', 'motorcycle', 
                        'bicycle', 'traffic_cone']

gt_info = (gt_boxes, simplied_class_names)

# tmp_data = dataset[0]
# print(tmp_data['img_metas'][0]['scene_token'])
# for key, value in tmp_data.items():
#     if "gt" in key:
#         print(key, value)

# img_tensor = get_img_tensor(tmp_data)
# print(img_tensor.shape) # img_tensor.shape = [1, 6, 3, 928, 1600]
# print(img_tensor.squeeze()[1].shape)

# print(img_tensor.squeeze()[1].min())
# print(img_tensor.squeeze()[1].max())
# img_tensor_1 = normlise_img(img_tensor.squeeze()[1])
# print(img_tensor_1.max())
# print(img_tensor_1.min())
# save_image(img_tensor.squeeze()[1], './verification/test.png')
# save_image(img_tensor_1, './verification/test_1.png')


query_model = copy.deepcopy(model)

for i, data in enumerate(data_loader):

    task = OpticalVerification(query_model,
                               copy.deepcopy(model.module),
                               data,
                               gt_info,
                               negative_dist)
    object_func = task.set_problem()

    direct_solver = LowBoundedDIRECT_full_parrallel(object_func,len(bound), bound, 
                                             args.max_iteration, 
                                             args.max_deep, 
                                             args.max_evaluation, 
                                             args.tolerance,
                                             debug=False)
    start_time = time.time()
    direct_solver.solve()
    end_time = time.time()
    print(f"{(end_time-start_time)/60:.2f} min")

    # print(f'##################### Start query {i} #####################')
    # for _ in range(3): 
    #     tmp_data = copy.deepcopy(data)
    #     query_model.module = copy.deepcopy(model.module)
    #     with torch.no_grad():
    #         query_result = query_model(return_loss=False, rescale=True, **tmp_data)
    #     _dist = results2dist(query_result[0],
    #                         gt_boxes,
    #                         simplied_class_names, verbose=True)
    #     print(_dist)
    # print(f'#####################  End query {i}  #####################')
    ori_img = OpticalVerification.get_img_tensor(data)
    # ori_transf = np.array([0, 1., 1.]*6)
    tmp_img = OpticalVerification.functional_perturb(ori_img,direct_solver.optimal_result())
    tmp_data = OpticalVerification.replace_img_tensor(data, tmp_img.unsqueeze(0))

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **tmp_data)
    _dist = negative_dist(result[0],
                            gt_boxes,
                            simplied_class_names, verbose=True)
    print(_dist)

    # query_model.module.prev_frame_info = copy.deepcopy(model.module.prev_frame_info)
    # query_model.module.prev_bev = copy.deepcopy(model.module.prev_bev)

    if i > 2: break


####################################################
#            
####################################################
# img = copy.deepcopy(get_img_tensor(tmp_data)).squeeze()
# trans_img = optical_transform(img, [0.1]*4)
# # print(get_img_tensor(tmp_data).shape)
# # for i, data in enumerate(data_loader):
# #     with torch.no_grad():
# #         result = model(return_loss=False, rescale=True, **data)
# trans_data = copy.deepcopy(tmp_data)
# trans_data['img'][0].data[0] = trans_img


####################################################
#            
####################################################
# print("Loaded results from {}. Found detections for {} samples."
#               .format(sample_token, len(all_results.sample_tokens)))
# for box in boxes:
#     print(box.ego_dist)
#     break
# sample_rec = dataset.nusc.get('sample', sample_token)
# sd_record = dataset.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
# pose_record = dataset.nusc.get('ego_pose', sd_record['ego_pose_token'])
# print(pose_record['translation'])

# print(type(result[0]['boxes_3d']))
# print(result[0]['labels_3d'].shape)
# # print(dir(result[0]['boxes_3d']))
# print(result[0]['boxes_3d'].tensor.shape)
# print(result[0]['boxes_3d'].gravity_center.numpy().shape)

####################################################
#            
####################################################

# results = []
# for i, data in enumerate(data_loader):
#     with torch.no_grad():
#         result = model(return_loss=False, rescale=True, **data)
#     results.extend(result)

# kwargs = {} if args.eval_options is None else args.eval_options
# kwargs['jsonfile_prefix'] = osp.join('test', args.config.split(
#     '/')[-1].split('.')[-2], time.ctime().replace(' ', '_').replace(':', '_'))
# eval_kwargs = cfg.get('evaluation', {}).copy()
# # hard-code way to remove EvalHook args
# for key in [
#         'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
#         'rule'
# ]:
#     eval_kwargs.pop(key, None)
# eval_kwargs.update(dict(metric='bbox', **kwargs))
# final_out = dataset.evaluate(results, **eval_kwargs)
# print(type(final_out))
# print(final_out['pts_bbox_NuScenes/NDS'])

####################################################
####################################################

# {'pipeline': [{'type': 'LoadMultiViewImageFromFilesInCeph', 'to_float32': True, 'file_client_args': {'backend': 'disk'}, 'img_root': ''}, {'type': 'NormalizeMultiviewImage', 'mean': [103.53, 116.28, 123.675], 'std': [1.0, 1.0, 1.0], 'to_rgb': False}, {'type': 'PadMultiViewImage', 'size_divisor': 32}, {'type': 'LoadAnnotations3D_E2E', 'with_bbox_3d': False, 'with_label_3d': False, 'with_attr_label': False, 'with_future_anns': True, 'with_ins_inds_3d': False, 'ins_inds_add_1': True}, {'type': 'GenerateOccFlowLabels', 'grid_conf': {'xbound': [-50.0, 50.0, 0.5], 'ybound': [-50.0, 50.0, 0.5], 'zbound': [-10.0, 10.0, 20.0]}, 'ignore_index': 255, 'only_vehicle': True, 'filter_invisible': False}, {'type': 'MultiScaleFlipAug3D', 'img_scale': (1600, 900), 'pts_scale_ratio': 1, 'flip': False, 'transforms': [{'type': 'DefaultFormatBundle3D', 'class_names': ['car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'], 'with_label': False}, {'type': 'CustomCollect3D', 'keys': ['img', 'timestamp', 'l2g_r_mat', 'l2g_t', 'gt_lane_labels', 'gt_lane_bboxes', 'gt_lane_masks', 'gt_segmentation', 'gt_instance', 'gt_centerness', 'gt_offset', 'gt_flow', 'gt_backward_flow', 'gt_occ_has_invalid_frame', 'gt_occ_img_is_valid', 'sdc_planning', 'sdc_planning_mask', 'command']}]}], 'metric': 'bbox', 'jsonfile_prefix': 'test/base_track_map/Mon_Aug__7_17_40_44_2023'}