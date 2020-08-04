import math

from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import torch



running_loss_final = 0
running_loss_model2 = 0
train_pre = 0




def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()

    input = input.transpose(1,2).transpose(2,3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    loss = F.cross_entropy(input, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss



class Trainer(object):

    def __init__(self, cuda, model_rgb,model_depth,model_fusion, optimizer_rgb,
                 optimizer_depth,optimizer_fusion,
                 train_loader, max_iter, snapshot, outpath, sshow, size_average=False):
        self.cuda = cuda
        self.model_rgb = model_rgb
        self.model_depth = model_depth
        self.model_fusion = model_fusion
        self.optim_rgb = optimizer_rgb
        self.optim_depth = optimizer_depth
        self.optim_fusion = optimizer_fusion
        self.train_loader = train_loader
        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.snapshot = snapshot
        self.outpath = outpath
        self.sshow = sshow
        self.size_average = size_average



    def train_epoch(self):

        for batch_idx, (data, target, depth) in enumerate(self.train_loader):


            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue # for resuming
            self.iteration = iteration
            if self.iteration >= self.max_iter:
                break
            if self.cuda:
                data, target, depth = data.cuda(), target.cuda(), depth.cuda()
            data, target, depth = Variable(data), Variable(target), Variable(depth)
            n, c, h, w = data.size()        # batch_size, channels, height, weight
            depth = depth.view(n,h,w,1).repeat(1,1,1,c)
            depth = depth.transpose(3,1)
            depth = depth.transpose(3,2)



            self.optim_rgb.zero_grad()
            self.optim_depth.zero_grad()
            self.optim_fusion.zero_grad()

            global running_loss_final
            global running_loss_model2


            h1,h2,h3,h4,h5 = self.model_rgb(data)       # RGBNet's output
            d1,d2,d3,d4,d5 = self.model_depth(depth)    # DepthNet's output

            # ------------------------------ Fusion --------------------------- #
            output1,output2,output3,output4,output5,output = self.model_fusion(h1,h2,h3,h4,h5,d1,d2,d3,d4,d5)     # Final output


            loss1 = cross_entropy2d(output1, target, size_average=self.size_average)
            loss2 = cross_entropy2d(output2, target, size_average=self.size_average)
            loss3 = cross_entropy2d(output3, target, size_average=self.size_average)
            loss4 = cross_entropy2d(output4, target, size_average=self.size_average)
            loss5 = cross_entropy2d(output5, target, size_average=self.size_average)
            loss = cross_entropy2d(output, target, size_average=self.size_average)



            loss_all= loss1+loss2+loss3+loss4+loss5+loss
            loss_model2 = loss

            running_loss_final += loss_all.item()
            running_loss_model2 += loss_model2.item()            




            if iteration % self.sshow == (self.sshow-1):
                print('\n [%3d, %6d,   RGB-D Net loss: %.3f,  model2 loss: %.3f]' % (self.epoch + 1, iteration + 1, running_loss_final / (n * self.sshow), running_loss_model2 / (n * self.sshow)))

                running_loss_final = 0.0
                running_loss_model2 = 0.0



            if iteration <= 200000:
                if iteration % self.snapshot == (self.snapshot-1):
                #if iteration % 100 == (100 - 1):
                    savename = ('%s/snapshot_iter_%d.pth' % (self.outpath, iteration+1+train_pre))
                    torch.save(self.model_rgb.state_dict(), savename)
                    print('save: (snapshot: %d)' % (iteration+1+train_pre))
                
                    savename_focal = ('%s/depth_snapshot_iter_%d.pth' % (self.outpath, iteration+1+train_pre))
                    torch.save(self.model_depth.state_dict(), savename_focal)
                    print('save: (snapshot_depth: %d)' % (iteration+1+train_pre))

                    savename_fusion = ('%s/fusion_snapshot_iter_%d.pth' % (self.outpath, iteration+1+train_pre))
                    torch.save(self.model_fusion.state_dict(), savename_fusion)
                    print('save: (snapshot_fusion: %d)' % (iteration+1+train_pre))

            else:
                if iteration % 10000 == (10000 - 1):
                    savename = ('%s/snapshot_iter_%d.pth' % (self.outpath, iteration+1+train_pre))
                    torch.save(self.model_rgb.state_dict(), savename)
                    print('save: (snapshot: %d)' % (iteration + 1 + train_pre))

                    savename_focal = ('%s/depth_snapshot_iter_%d.pth' % (self.outpath, iteration+1+train_pre))
                    torch.save(self.model_depth.state_dict(), savename_focal)
                    print('save: (snapshot_depth: %d)' % (iteration + 1 + train_pre))

                    savename_fusion = ('%s/fusion_snapshot_iter_%d.pth' % (self.outpath, iteration+1+train_pre))
                    torch.save(self.model_fusion.state_dict(), savename_fusion)
                    print('save: (snapshot_fusion: %d)' % (iteration + 1 + train_pre))



            if (iteration+1) == self.max_iter:
                savename = ('%s/snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                torch.save(self.model_rgb.state_dict(), savename)
                print('save: (snapshot: %d)' % (iteration+1))

                savename_focal = ('%s/depth_snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                torch.save(self.model_depth.state_dict(), savename_focal)
                print('save: (snapshot_depth: %d)' % (iteration+1))

                savename_fusion = ('%s/fusion_snapshot_iter_%d.pth' % (self.outpath, iteration+1))
                torch.save(self.model_fusion.state_dict(), savename_fusion)
                print('save: (snapshot_fusion: %d)' % (iteration+1))



            loss_all.backward(retain_graph = True)
            self.optim_fusion.step()
            self.optim_depth.step()
            self.optim_rgb.step()

            loss_model2.backward()
            self.optim_fusion.step()
            self.optim_depth.step()
            self.optim_rgb.step()


    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))

        for epoch in range(max_epoch):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
