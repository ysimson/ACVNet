# from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss_train_attn_only, model_loss_train_freeze_attn, model_loss_train, \
    model_loss_test
from utils import *
from torch.utils.data import DataLoader
from collections import OrderedDict
import matplotlib.pyplot as plt
import gc
# from apex import amp
import cv2


def test():
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '0,1,2,3'
    device = 'cpu'

    parser = argparse.ArgumentParser(
        description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
    parser.add_argument('--model', default='acvnet', help='select a model structure', choices=__models__.keys())
    parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
    parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
    parser.add_argument('--loadckpt', default='./pretrained_model/sceneflow.ckpt',
                        help='load the weights from a specific checkpoint')
    parser.add_argument('-l', '--left_image_fn', help='Filename of left image', required=True)
    parser.add_argument('-r', '--right_image_fn', help='Filename of left image', required=True)
    parser.add_argument('-o', '--output_directory', help="Directory to save output", default="demo_output")
    args = parser.parse_args()

    left_img = cv2.imread(args.left_image_fn)
    right_img = cv2.imread(args.right_image_fn)

    # model, optimizer
    model = __models__[args.model](args.maxdisp, False, False)
    # model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    # load parameters
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)

    model_state_dict = OrderedDict()
    for k, v in state_dict['model'].items():
        name = k[7:]  # remove `module.`
        model_state_dict[name] = v
    model.load_state_dict(model_state_dict)  # state_dict['model'])

    imgL_ = left_img.transpose(2, 0, 1)
    imgR_ = right_img.transpose(2, 0, 1)
    imgL_ = np.ascontiguousarray(imgL_[None, :, :, :])
    imgR_ = np.ascontiguousarray(imgR_[None, :, :, :])

    imgL = torch.tensor(imgL_.astype("float32")).to(device)
    imgR = torch.tensor(imgR_.astype("float32")).to(device)

    disp_ests = model(imgL, imgR)

    disparity = disp_ests[0].cpu().detach().numpy()[0, :, :]
    disparity_gt = disp_ests[0].cpu().detach().numpy()[0, :, :]
    fig, axes = plt.subplots(1, 2, figsize=(12, 12))
    axes[0].imshow(disparity)
    axes[0].set_title("disparity est")

    left = imgL.cpu().numpy()[0, :, :, :]
    left = np.transpose(left,  (1, 2, 0))
    axes[1].imshow(left)
    axes[1].set_title("left")

    plt.show()


if __name__ == '__main__':
    test()
