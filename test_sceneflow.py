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

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # '0,1,2,3'
device = 'cpu'

parser = argparse.ArgumentParser(
    description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
parser.add_argument('--model', default='acvnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="/data/sceneflow/", help='data path')
parser.add_argument('--testlist', default='./filenames/sceneflow_test.txt', help='testing list')
parser.add_argument('--test_batch_size', type=int, default=8, help='testing batch size')
parser.add_argument('--loadckpt', default='./pretrained_model/sceneflow.ckpt',
                    help='load the weights from a specific checkpoint')

# parse arguments, set seeds
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=16, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp, False, False)
# model = nn.DataParallel(model)
model.to(device)

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)

model_state_dict = OrderedDict()
for k, v in state_dict['model'].items():
    name = k[7:]  # remove `module.`
    model_state_dict[name] = v
model.load_state_dict(model_state_dict)  #state_dict['model'])


def test():
    avg_test_scalars = AverageMeterDict()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, dip_est = test_sample(sample)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs
        print('Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(batch_idx,
                                                                    len(TestImgLoader), loss,
                                                                    time.time() - start_time))

    avg_test_scalars = avg_test_scalars.mean()
    print("avg_test_scalars", avg_test_scalars)


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = torch.tensor(imgL).to(device)
    imgR = torch.tensor(imgR).to(device)
    disp_gt = torch.tensor(disp_gt).to(device)
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    disp_ests = model(imgL, imgR)
    disp_gts = [disp_gt]
    loss = model_loss_test(disp_ests, disp_gt, mask)
    scalar_outputs = {"loss": loss}
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    disparity = disp_ests[0].cpu().numpy()[0, :, :]
    disparity_gt = disp_gt.cpu().numpy()[0, :, :]
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0, 0].imshow(disparity)
    axes[0, 0].set_title("disparity est")

    axes[0, 1].imshow(disparity_gt)
    axes[0, 1].set_title("disparity gt")

    left = imgL.cpu().numpy()[0, :, :, :]
    left = np.transpose(left,  (1, 2, 0))
    axes[1, 0].imshow(left)
    axes[1, 0].set_title("left")

    occ_mask = mask[0, :, :]
    axes[1, 1].imshow(occ_mask)
    axes[1, 1].set_title("occ_mask")

    return tensor2float(loss), tensor2float(scalar_outputs), disp_ests


if __name__ == '__main__':
    test()
