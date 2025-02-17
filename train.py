'''
https://github.com/skylook/SuperGlue/blob/master/train.py
'''

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn

# Datasets
from dataset import *

import os
import torch.multiprocessing
from tqdm import tqdm

# torch.backends.cudnn.benchmark = True

# from models.matching import Matching
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics)

from models.superpoint import SuperPoint
from models.superglue import SuperGlue
from test import validation
import wandb

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# torch.multiprocessing.set_start_method("spawn")
# torch.cuda.set_device(0)

# try:
#     torch.multiprocessing.set_start_method('spawn')
# except RuntimeError:
#     pass


parser = argparse.ArgumentParser(
    description='Image pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    '--exp_name', type=str,  
    help='name_of_the_experiment')
parser.add_argument(
    '--viz', action='store_true',
    help='Visualize the matches and dump the plots')
parser.add_argument(
    '--eval', action='store_true',
    help='Perform the evaluation'
            ' (requires ground truth pose and intrinsics)')

parser.add_argument(
    '--detector', choices={'superpoint', 'sift'}, default='superpoint',
    help='Keypoint detector')
parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    help='SuperGlue weights')
parser.add_argument(
    '--max_keypoints', type=int, default=1024,
    help='Maximum number of keypoints detected by Superpoint'
            ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.005,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--nms_radius', type=int, default=4,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
    ' (Must be positive)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')

parser.add_argument(
    '--resize', type=int, nargs='+', default=[640, 480],
    help='Resize the input image before running inference. If two numbers, '
            'resize to the exact dimensions, if one number, resize the max '
            'dimension, if -1, do not resize')
parser.add_argument(
    '--resize_float', action='store_true',
    help='Resize the image after casting uint8 to float')

parser.add_argument(
    '--cache', action='store_true',
    help='Skip the pair if output .npz files are already found')
parser.add_argument(
    '--show_keypoints', action='store_true',
    help='Plot the keypoints in addition to the matches')
parser.add_argument(
    '--fast_viz', action='store_true',
    help='Use faster image visualization based on OpenCV instead of Matplotlib')
parser.add_argument(
    '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
    help='Visualization file extension. Use pdf for highest-quality.')

parser.add_argument(
    '--opencv_display', action='store_true',
    help='Visualize via OpenCV before saving output images')
parser.add_argument(
    '--eval_pairs_list', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
    help='Path to the list of image pairs for evaluation')
parser.add_argument(
    '--shuffle', action='store_true',
    help='Shuffle ordering of pairs before processing')
parser.add_argument(
    '--max_length', type=int, default=-1,
    help='Maximum number of pairs to evaluate')

parser.add_argument(
    '--eval_input_dir', type=str, default='assets/scannet_sample_images/',
    help='Path to the directory that contains the images')
parser.add_argument(
    '--eval_output_dir', type=str, default='dump_match_pairs/',
    help='Path to the directory in which the .npz results and optional,'
            'visualizations are written')
parser.add_argument(
    '--learning_rate', type=int, default=0.0001,
    help='Learning rate')

    
parser.add_argument(
    '--batch_size', type=int, default=1,
    help='batch_size')
parser.add_argument(
    '--train_path', type=str, default='assets/freiburg_sequence', # MSCOCO2014_yingxin
    help='Path to the directory of training imgs.')
# parser.add_argument(
#     '--nfeatures', type=int, default=1024,
#     help='Number of feature points to be extracted initially, in each img.')
parser.add_argument(
    '--epoch', type=int, default=20,
    help='Number of epoches')



if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)

    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'



    # store viz results
    eval_output_dir = Path(opt.eval_output_dir)
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    print('Will write visualization images to',
        'directory \"{}\"'.format(eval_output_dir))

    # detector_factory = {
    #     'superpoint': SuperPointDataset,
    #     'sift': SIFTDataset,
    # }
    detector_dims = {
        'superpoint': 256,
        'sift': 128,
    }

    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints,
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
            'descriptor_dim': detector_dims[opt.detector],
        }
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set data loader
    if opt.detector == 'superpoint':
        train_set = SuperPointDataset("", image_list='datasets/train.txt', device='cuda', max_keypoints=opt.max_keypoints)
    else:
        RuntimeError('Error detector : {}'.format(opt.detector))

    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=True)

    superglue = SuperGlue(config.get('superglue', {}))
    superglue.load_state_dict(torch.load("models/weights/superglue_outdoor.pth"))

    if torch.cuda.is_available():
        superglue.cuda()
    else:
        print("### CUDA not available ###")

    optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)

    mean_loss = []
    for epoch in range(1, opt.epoch+1):
        epoch_loss = 0
        superglue.train()
        # train_loader = tqdm(train_loader)
        for i, data in enumerate(train_loader):
            data['is_train'] = True
            pred = superglue(data)
            pred = {**pred, **data}

            if pred['skip_train'] == True: # image has no keypoint
                continue

            optimizer.zero_grad()
            Loss = pred['loss']
            epoch_loss += Loss.item()
            mean_loss.append(Loss.item()) 
            # Loss.backward()
            # optimizer.step()

            torch.cuda.empty_cache()

            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch, opt.epoch, i+1, len(train_loader), np.mean(mean_loss)))   # Loss.item()    
                mean_loss = []

                # Visualize the matches.
                image0, image1 = pred['image0'].cpu().numpy()[0]*255., pred['image1'].cpu().numpy()[0]*255.
                kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
                matches, conf = pred['matches0'].cpu().detach().numpy()[0], pred['matching_scores0'].cpu().detach().numpy()[0]
                valid = matches > -1
                mkpts0 = kpts0[valid]
                mkpts1 = kpts1[matches[valid]]
                mconf = conf[valid]
                viz_path = eval_output_dir / '{}_matches.{}'.format(str(i), opt.viz_extension)
                color = cm.jet(mconf)
                stem = pred['file_name']
                text = []

                out = make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1, color, text, viz_path, stem, opt.show_keypoints,opt.fast_viz, opt.opencv_display, 'Matches')

        ### eval ###
        superglue.eval()
        homogrpahy_auc = validation(superglue, 'datasets/val.txt')

        epoch_loss /= len(train_loader)

        if not os.path.isdir("exp"):
            os.makedirs("exp")

        model_out_path = "exp/model_epoch_{}_{}.pth".format(epoch, homogrpahy_auc)
        torch.save(superglue, model_out_path)
        print("Epoch [{}/{}] done. Epoch Loss {}. Checkpoint saved to {}"
            .format(epoch, opt.epoch, epoch_loss, model_out_path))
        