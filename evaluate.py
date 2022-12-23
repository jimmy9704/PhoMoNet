import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

import model_io
from dataloader_spad import DepthDataLoader
from models import UnetAdaptiveBins
from utils import RunningAverageDict

import matplotlib.pyplot as plt
import matplotlib
import time

def colorize(arr, vmin=0.1, vmax=10, cmap='magma_r', ignore=-1):
    invalid_mask = arr == ignore

    # normalize
    vmin = arr.min() if vmin is None else vmin
    vmax = arr.max() if vmax is None else vmax
    if vmin != vmax:
        arr = (arr - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        arr = arr * 0.
    cmapper = matplotlib.cm.get_cmap(cmap)
    arr = cmapper(arr, bytes=True)  # (nxmx4)
    arr[invalid_mask] = 255
    img = arr[:, :, :3]

    return img

def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return dict(a1=a1, a2=a2, a3=a3, abs_rel=abs_rel, rmse=rmse, log_10=log_10, rmse_log=rmse_log,
                silog=silog, sq_rel=sq_rel)

def predict_tta(model, image, spad, args):
    pred = model(image,spad)[1]
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)
    final = nn.functional.interpolate(torch.Tensor(pred), image.shape[-2:], mode='bilinear', align_corners=True)
    return torch.Tensor(final)


def eval(model, test_loader, args, gpus=None, ):
    if gpus is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = gpus[0]

    if args.save_dir is not None:
        if os.path.isdir(args.save_dir) == False:
            os.makedirs(args.save_dir)

    metrics = RunningAverageDict()
    total_invalid = 0
    with torch.no_grad():
        model.eval()

        sequential = test_loader
        for batch in tqdm(sequential):

            image = batch['image'].to(device)
            gt = batch['depth'].to(device)
            spad = batch['spad'].to(device)

            w = int(image.shape[2]*1)
            h = int(image.shape[3]*1)
            image = nn.functional.interpolate(image, [w,h], mode='bilinear', align_corners=True)
            gt = nn.functional.interpolate(gt.permute(0,3,1,2), [w,h], mode='bilinear', align_corners=True).permute(0,2,3,1)

            final = predict_tta(model, image, spad, args)
            final = final.squeeze().cpu().numpy()

            final[np.isinf(final)] = args.max_depth
            final[np.isnan(final)] = args.min_depth

            if args.save_dir is not None:
                if args.dataset == 'nyu':
                    impath = f"{batch['image_path'][0].replace('/', '__').replace('.jpg', '')}"
                    factor = 1000
                else:
                    dpath = batch['image_path'][0].split('/')
                    impath = dpath[1] + "_" + dpath[-1]
                    impath = impath.split('.')[0]
                    factor = 256

                pred_path = os.path.join(args.save_dir, f"{impath}.png")
                gt_img = gt.squeeze().cpu().numpy() 
                plt.imsave(pred_path, colorize(final), cmap='jet')
                Image.fromarray(colorize(gt_img,cmap='jet')).save(pred_path)

            if 'has_valid_depth' in batch:
                if not batch['has_valid_depth']:
                    # print("Invalid ground truth")
                    total_invalid += 1
                    continue

            gt = gt.squeeze().cpu().numpy()
            valid_mask = np.logical_and(gt > args.min_depth, gt < args.max_depth)
            if args.garg_crop or args.eigen_crop:
                gt_height, gt_width = gt.shape
                eval_mask = np.zeros(valid_mask.shape)

                if args.garg_crop:
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
                valid_mask = np.logical_and(valid_mask, eval_mask)
            #             gt = gt[valid_mask]
            #             final = final[valid_mask]

            metrics.update(compute_errors(gt[valid_mask], final[valid_mask]))

    print(f"Total invalid: {total_invalid}")
    metrics = {k: round(v, 3) for k, v in metrics.get_value().items()}
    print(f"Metrics: {metrics}")


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield str(arg)


if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser(description='Model evaluator', fromfile_prefix_chars='@',
                                     conflict_handler='resolve')
    parser.convert_arg_line_to_args = convert_arg_line_to_args
    parser.add_argument('--n-bins', '--n_bins', default=256, type=int,
                        help='number of bins/buckets to divide depth range into')
    parser.add_argument('--gpu', default=None, type=int, help='Which gpu to use')
    parser.add_argument('--save-dir', '--save_dir', default=None, type=str, help='Store predictions in folder')
    parser.add_argument("--root", default=".", type=str,
                        help="Root folder to save data in")

    parser.add_argument("--dataset", default='nyu', type=str, help="Dataset to train on")

    parser.add_argument("--data_path", default='../../dataset/nyu/sync/', type=str,
                        help="path to dataset")
    parser.add_argument("--gt_path", default='../../dataset/nyu/sync/', type=str,
                        help="path to dataset gt")

    parser.add_argument('--filenames_file',
                        default="./train_test_inputs/nyudepthv2_train_files_with_gt.txt",
                        type=str, help='path to the filenames text file')

    parser.add_argument('--input_height', type=int, help='input height', default=416)
    parser.add_argument('--input_width', type=int, help='input width', default=544)
    parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
    parser.add_argument('--min_depth', type=float, help='minimum depth in estimation', default=1e-3)

    parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')

    parser.add_argument('--data_path_eval',
                        default="../../dataset/nyu/official_splits/test/",
                        type=str, help='path to the data for online evaluation')
    parser.add_argument('--gt_path_eval', default="../../dataset/nyu/official_splits/test/",
                        type=str, help='path to the groundtruth data for online evaluation')
    parser.add_argument('--filenames_file_eval',
                        default="./train_test_inputs/nyudepthv2_test_files_with_gt.txt",
                        type=str, help='path to the filenames text file for online evaluation')
    parser.add_argument('--checkpoint_path', '--checkpoint-path', type=str, required=True,
                        help="checkpoint file to use for prediction")

    parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
    parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=10)
    parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_false')
    parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
    parser.add_argument('--do_kb_crop', help='Use kitti benchmark cropping', action='store_true')
    ################################SPAD######################################################################
    parser.add_argument('--signal_count', type=float, help='number os spad signal_count', default=1e6)
    parser.add_argument('--sbr', type=float, help='spad sbr', default=100)
    parser.add_argument('--spad_bins', type=float, help='spad time bin count', default=1024)
    parser.add_argument('--laser_fwhm_ps', type=float, help='laser_fwhm_ps', default=70)

    if sys.argv.__len__() == 2:
        arg_filename_with_prefix = '@' + sys.argv[1]
        args = parser.parse_args([arg_filename_with_prefix])
    else:
        args = parser.parse_args()

    # args = parser.parse_args()
    args.gpu = int(args.gpu) if args.gpu is not None else 0
    args.distributed = False
    device = torch.device('cuda:{}'.format(args.gpu))
    test = DepthDataLoader(args, 'online_eval').data
    model = UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                   norm='linear').to(device)
    model = model_io.load_checkpoint(args.checkpoint_path, model)[0]
    model = model.eval()

    eval(model, test, args, gpus=[device])
