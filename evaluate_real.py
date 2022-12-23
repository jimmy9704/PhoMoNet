import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

import model_io
from dataloader_captured import CapturedDataLoader
from models import UnetAdaptiveBins
from utils import RunningAverageDict

import matplotlib.pyplot as plt
import time

import matplotlib

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


# def denormalize(x, device='cpu'):
#     mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
#     std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
#     return x * std + mean
#
def predict_tta(model, image, spad, args):
    img = nn.functional.interpolate(image, [480,640], mode='bilinear', align_corners=True)
    pred = model(img,spad)[1]
    #     pred = utils.depth_norm(pred)
    #     pred = nn.functional.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)
    #     pred = np.clip(pred.cpu().numpy(), 10, 1000)/100.
    pred = np.clip(pred.cpu().numpy(), args.min_depth, args.max_depth)

    image = torch.Tensor(np.array(image.cpu().numpy())[..., ::-1].copy()).to(device)

    img = nn.functional.interpolate(image, [480,640], mode='bilinear', align_corners=True)
    # pred_lr = model(img,spad)[1]
    # #     pred_lr = utils.depth_norm(pred_lr)
    # #     pred_lr = nn.functional.interpolate(pred_lr, depth.shape[-2:], mode='bilinear', align_corners=True)
    # #     pred_lr = np.clip(pred_lr.cpu().numpy()[...,::-1], 10, 1000)/100.
    # pred_lr = np.clip(pred_lr.cpu().numpy()[..., ::-1], args.min_depth, args.max_depth)
    # final = 0.5 * (pred + pred_lr)
    final = pred
    final = nn.functional.interpolate(torch.Tensor(final), image.shape[-2:], mode='bilinear', align_corners=True)
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
    # crop_size = (471 - 45, 601 - 41)
    # bins = utils.get_bins(100)
    total_invalid = 0
    with torch.no_grad():
        model.eval()

        sequential = test_loader
        for batch in tqdm(sequential):

            image = batch['image'].to(device)
            gt = batch['depth'].to(device)
            spad = batch['spad'].to(device)
            final = predict_tta(model, image, spad, args)

            final = final.squeeze().cpu().numpy()
            #plt.imsave('test_rgb.png',\
            #image.cpu().squeeze().permute(1,2,0).numpy()*255)
            # plt.imsave('test_pred.png',final/10, cmap='magma_r')
            # time.sleep(10)

            # final[final < args.min_depth] = args.min_depth
            # final[final > args.max_depth] = args.max_depth
            final[np.isinf(final)] = args.max_depth
            final[np.isnan(final)] = args.min_depth

            if args.save_dir is not None:
                if args.dataset == 'nyu':
                    impath = f"{batch['image_path'][0][6:].replace('/', '__').replace('.jpg', '')}"
                    factor = 0.1
                else:
                    dpath = batch['image_path'][0].split('/')
                    impath = dpath[1] + "_" + dpath[-1]
                    impath = impath.split('.')[0]
                    factor = 256

                # rgb_path = os.path.join(rgb_dir, f"{impath}.png")
                # tf.ToPILImage()(denormalize(image.squeeze().unsqueeze(0).cpu()).squeeze()).save(rgb_path)

                pred_path = os.path.join(args.save_dir, f"{impath}.png")
                # pred = (final * factor)#.astype('uint8')
                # gt_img = gt.squeeze().cpu().numpy() 
                Image.fromarray(colorize(final ,cmap='jet')).save(pred_path)

                # #print(bin_edges)
                # valid_mask = np.logical_and(gt.cpu().numpy() > args.min_depth, gt.cpu().numpy() < args.max_depth)
                # final_hist, bin_edges = np.histogram(final[valid_mask[0,0]],bins=1024,range=(0,10))

                # gt_hist, bin_edges = np.histogram(gt_img[valid_mask[0,0]],bins=1024,range=(0,10))
                # fig, ax = plt.subplots()
                # ax.bar(bin_edges[1:], -1*final_hist, alpha=0.5,width=0.0095)
                # ax.bar(bin_edges[1:], gt_hist, alpha=0.5,width=0.0095)
                # ax.bar(bin_edges[1:], spad.cpu().numpy()[0]*gt_hist.max(), alpha=0.5,width=0.0095)
                # ax.set_xlim(0,10)
                # #ax.set_ylim(-1.3*gt_hist.max(),1.3*gt_hist.max())
                # ax.get_yaxis().set_visible(False)
                # ax.spines['left'].set_visible(False)   
                # ax.spines['right'].set_visible(False)          # 오른쪽 축을 보이지 않도록
                # ax.spines['top'].set_visible(False)            # 위 축을 보이지 않도록
                # ax.spines['bottom'].set_position(('data', 0))   # 아래 축을 데이터 0의 위치로 이동
                
                # plt_name = 'plot' + impath +'.png'
                # plt.savefig(plt_name)
                # plt.clf()

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
                    print("garg")
                    eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                    int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

                elif args.eigen_crop:
                    print("eigen")
                    if args.dataset == 'kitti':
                        eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                        int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
                    else:
                        eval_mask[45:471, 41:601] = 1
                valid_mask = np.logical_and(valid_mask, eval_mask)
            #             gt = gt[valid_mask]
            #             final = final[valid_mask]

            print(batch['file_name'])
            # initializing K 
            K = 3
            # loop to iterate for values 
            res = dict()
            test_dict = compute_errors(gt[valid_mask], final[valid_mask])
            for key in test_dict:
                # rounding to K using round()
                res[key] = round(test_dict[key], K)
            print(res)
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

    parser.add_argument('--input_height', type=int, help='input height', default=480)
    parser.add_argument('--input_width', type=int, help='input width', default=640)
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
    test = CapturedDataLoader('../../dataset/captured/processed').data
    model = UnetAdaptiveBins.build(n_bins=args.n_bins, min_val=args.min_depth, max_val=args.max_depth,
                                   norm='linear').to(device)
    model = model_io.load_checkpoint(args.checkpoint_path, model)[0]
    model = model.eval()

    eval(model, test, args, gpus=[device])
