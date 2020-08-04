
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision

import torch.nn.functional as F
import torch.optim as optim
from dataset_loader import MyData, MyTestData
from model import RGBNet,DepthNet
from fusion import Fusion
from functions import imsave
import argparse
from trainer import Trainer
from refiner import Refiner
from utils.evaluateFM import get_FM
import time

import os

sss=0
configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=1000000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        spshot=10000,
        nclass=2,
        sshow=10,
    )
}

parser=argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='test', help='train or test')     # train test refine
parser.add_argument('--param', type=str, default=True, help='path to pre-trained parameters')

"train database"
   # train
parser.add_argument('--train_dataroot', type=str, default='path of your train data ', help='path to train data')

"test database"
   # DUT_RGBD
parser.add_argument('--test_dataroot', type=str, default='path of your test data ', help='path to test data')

parser.add_argument('--snapshot_root', type=str, default='path of your checkpoint', help='path to snapshot')
parser.add_argument('--salmap_root', type=str, default='path of your result', help='path to saliency map')
parser.add_argument('-c', '--config', type=int, default=1, choices=configurations.keys())
args = parser.parse_args()
cfg = configurations[args.config]

cuda = torch.cuda.is_available



"""""""""""~~~ dataset loader ~~~"""""""""

train_dataRoot = args.train_dataroot
test_dataRoot = args.test_dataroot


if not os.path.exists(args.snapshot_root):
    os.mkdir(args.snapshot_root)
if not os.path.exists(args.salmap_root):
    os.mkdir(args.salmap_root)


if args.phase == 'train':
    SnapRoot = args.snapshot_root           # checkpoint
    train_loader = torch.utils.data.DataLoader(MyData(train_dataRoot, transform=True),
                                               batch_size=2, shuffle=True, num_workers=4, pin_memory=True)

if args.phase == 'test':
    MapRoot = args.salmap_root
    test_loader = torch.utils.data.DataLoader(MyTestData(test_dataRoot, transform=True),
                                   batch_size=1, shuffle=True, num_workers=4, pin_memory=True)
print ('data already')
""""""""""" ~~~nets~~~ """""""""

start_epoch = 0
start_iteration = 0


model_rgb = RGBNet(cfg['nclass'])
model_depth = DepthNet(cfg['nclass'])
model_fusion = Fusion(cfg['nclass'])





if args.param is True:
    model_rgb.load_state_dict(torch.load(os.path.join(args.snapshot_root, '.pth')))
    model_depth.load_state_dict(torch.load(os.path.join(args.snapshot_root, '.pth')))
    model_fusion.load_state_dict(torch.load(os.path.join(args.snapshot_root, '.pth')))

else:
    vgg19_bn = torchvision.models.vgg19_bn(pretrained=True)
    model_rgb.copy_params_from_vgg19_bn(vgg19_bn)
    model_depth.copy_params_from_vgg19_bn(vgg19_bn)


if cuda:
   model_rgb = model_rgb.cuda()
   model_depth = model_depth.cuda()
   model_fusion = model_fusion.cuda()








if args.phase == 'train':

    # Trainer: class, defined in trainer.py
    optimizer_rgb = optim.SGD(model_rgb.parameters(), lr=cfg['lr'],momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    optimizer_depth = optim.SGD(model_depth.parameters(), lr=cfg['lr'],momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    optimizer_fusion = optim.SGD(model_fusion.parameters(), lr=cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])

    training = Trainer(
        cuda=cuda,
        model_rgb=model_rgb,
        model_depth=model_depth,
        model_fusion=model_fusion,
        optimizer_rgb=optimizer_rgb,
        optimizer_depth=optimizer_depth,
        optimizer_fusion=optimizer_fusion,
        train_loader=train_loader,
        max_iter=cfg['max_iteration'],
        snapshot=cfg['spshot'],
        outpath=args.snapshot_root,
        sshow=cfg['sshow']
    )
    training.epoch = start_epoch
    training.iteration = start_iteration
    training.train()


if args.phase == 'test':


    for id, (data, depth, img_name, img_size) in enumerate(test_loader):
        print('testing bach %d' % id)

        inputs = Variable(data).cuda()
        inputs_depth = Variable(depth).cuda()

        n, c, h, w = inputs.size()
        depth = inputs_depth.view(n, h, w, 1).repeat(1, 1, 1, c)
        depth = depth.transpose(3, 1)
        depth = depth.transpose(3, 2)

        h1, h2, h3, h4, h5 = model_rgb(inputs)  # RGBNet's output
        d1, d2, d3, d4, d5 = model_depth(depth)  # DepthNet's output
        outputs_all = model_fusion(h1, h2, h3, h4, h5, d1, d2, d3, d4, d5)  # Final output


        outputs_all = F.softmax(outputs_all, dim=1)
        outputs = outputs_all[0][1]

        outputs = outputs.cpu().data.resize_(h, w)
        imsave(os.path.join(MapRoot, img_name[0] + '.png'), outputs, img_size)




    # -------------------------- validation --------------------------- #
    torch.cuda.empty_cache()

    print("\nevaluating mae....")
    F_measure, mae = get_FM(salpath=MapRoot+'/', gtpath=test_dataRoot+'/test_masks/')
    print('F_measure:', F_measure)
    print('MAE:', mae)

